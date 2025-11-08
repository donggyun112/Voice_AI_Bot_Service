import asyncio, json, uuid, zmq.asyncio, numpy as np, traceback, time, io
from typing import AsyncGenerator, Optional, Any, Dict, List, Tuple
from contextlib import asynccontextmanager

from pipecat.frames.frames import (
	Frame, TTSAudioRawFrame, TTSStartedFrame,
	TTSStoppedFrame, ErrorFrame, UserStartedSpeakingFrame
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from loguru import logger


class TTSPipecService(TTSService):
	DEFAULT_SR = 24_000
	DEFAULT_CHUNK_SIZE = 256
	SOCKET_TIMEOUT = 3.0  # 초 단위 소켓 타임아웃
	RECONNECT_DELAY = 1.0  # 재연결 지연시간
	MAX_RECONNECTS = 3     # 최대 재연결 시도 횟수
	WATCHDOG_INTERVAL = 5.0  # 워치독 확인 간격
	DEFAULT_SR = 24_000
	REQUEST_TIMEOUT = 5.0   # 요청 타임아웃
	MAX_REQUEST_RETRIES = 2  # 최대 요청 재시도 횟수

	def __init__(
		self,
		*,
		server_address: str = "121.135.134.82",
		command_port: int = 5555,
		audio_port: int = 5556,
		default_voice: str = "KR",
		sample_rate: int = DEFAULT_SR,
		sample_format: str = "int16",
		chunk_size: int = DEFAULT_CHUNK_SIZE,
		debug: bool = False,
		**kwargs
	):
		super().__init__(sample_rate=sample_rate, push_stop_frames=True, **kwargs)

		self._server_address, self._command_port, self._audio_port = (
			server_address, command_port, audio_port
		)
		self._default_voice, self._sample_format = default_voice, "int16"
		self._chunk_size, self._debug = chunk_size, debug
		self._settings = {
			"voice": default_voice,
			"speed": 1.0,
			"sample_format": "int16",
			"chunk_size": chunk_size,
		}

		self._ctx = None
		self._cmd_sock = None
		self._audio_sock = None
		self._recv_task = None
		self._job_id: Optional[str] = None
		self._remote_sr: Optional[int] = None
		self._available_voices: Dict[str, Any] = {}

		self._request_queue = asyncio.Queue()
		self._queue_processor_task = None
		self._active_generators: Dict[str, Dict[str, Any]] = {}
		self._cmd_lock = asyncio.Lock()          # _cmd_sock send/recv 보호
		self._active = True
		self._socket_healthy = True
		self._last_successful_cmd = 0

		self._audio_buffers: Dict[str, io.BytesIO] = {}
		self._ignored_msg_counts: Dict[str, int] = {}

		self._stats = {"chunks_received": 0, "bytes_received": 0, "start_time": None}
		self._cleanup_task = None
		self._watchdog_task = None
		self._reconnect_count = 0

	async def _setup_sockets(self):
		"""ZMQ 소켓 초기화 """
		try:
			if self._cmd_sock:
				self._cmd_sock.close()
			if self._audio_sock:
				self._audio_sock.close()
			
			if self._ctx:
				try:
					self._ctx.term()
				except Exception:
					pass
			
			self._ctx = zmq.asyncio.Context()
			
			self._cmd_sock = self._ctx.socket(zmq.REQ)
			self._cmd_sock.setsockopt(zmq.RCVTIMEO, int(self.SOCKET_TIMEOUT * 1000))
			self._cmd_sock.setsockopt(zmq.SNDTIMEO, int(self.SOCKET_TIMEOUT * 1000))
			self._cmd_sock.setsockopt(zmq.LINGER, 0)
			self._cmd_sock.connect(f"tcp://{self._server_address}:{self._command_port}")
			
			self._audio_sock = self._ctx.socket(zmq.PULL)
			self._audio_sock.setsockopt(zmq.RCVTIMEO, int(self.SOCKET_TIMEOUT * 1000))
			self._audio_sock.setsockopt(zmq.LINGER, 0)
			self._audio_sock.connect(f"tcp://{self._server_address}:{self._audio_port}")
			
			self._socket_healthy = True
			self._reconnect_count = 0
			self._last_successful_cmd = time.time()
			return True
		
		except Exception as e:
			self._socket_healthy = False
			return False

	async def _reset_sockets(self):
		"""소켓 문제 발생 시 모든 소켓 재설정"""
		if self._reconnect_count >= self.MAX_RECONNECTS:
			return False
		self._reconnect_count += 1
		self._force_complete_all_jobs("Socket reset - connection lost")
		await asyncio.sleep(self.RECONNECT_DELAY)
		success = await self._setup_sockets()
		
		if success:
			try:
				await self._fetch_voices()
				return True
			except Exception as e:
				return False
		return False

	@asynccontextmanager
	async def _safe_cmd_communication(self):
		if not self._socket_healthy:
			if not await self._reset_sockets():
				raise RuntimeError("Socket is unhealthy and reset failed")
		try:
			async with self._cmd_lock:
				yield self._cmd_sock
			self._last_successful_cmd = time.time()
		except (zmq.error.Again, zmq.error.ZMQError) as e:
			self._socket_healthy = False
			raise RuntimeError(f"ZMQ communication failed: {e}")

	async def start(self, frame):
		await super().start(frame)

		if not await self._setup_sockets():
			logger.error("client socket setup failed")
		
		self._recv_task = self.create_task(self._recv_audio())
		self._queue_processor_task = self.create_task(self._process_queue())
		self._cleanup_task = self.create_task(self._cleanup_stale_jobs())
		self._watchdog_task = self.create_task(self._socket_watchdog())

		try:
			await self._fetch_voices()
		except Exception as e:
			logger.error("could not fetch voices: %s", e)

	async def stop(self, frame):
		await super().stop(frame)
		self._active = False

		for t in (self._queue_processor_task, self._recv_task, self._cleanup_task, self._watchdog_task):
			if t: await self.cancel_task(t)

		self._force_complete_all_jobs("Service stopping")

		if self._cmd_sock: self._cmd_sock.close()
		if self._audio_sock: self._audio_sock.close()
		if self._ctx: self._ctx.term()

		self._audio_buffers.clear()

	async def _process_queue(self):
		logger.info("[CLIENT] queue processor running")
		while self._active:
			try:
				text, gen = await self._request_queue.get()
			except asyncio.CancelledError:
				break

			try:
				await self._handle_request(text, gen)
			except Exception as e:
				logger.exception(f"[CLIENT] Error handling request: {e}")
				try:
					await gen.push_frame(ErrorFrame(f"Internal error: {str(e)}"))
				except:
					pass
			finally:
				self._request_queue.task_done()

	async def _handle_request(self, text: str, gen: "FrameGenerator"):
		if not text.strip():
			await gen.push_frame(ErrorFrame("Empty text"))
			return

		job_id = str(uuid.uuid4())
		self._job_id = job_id
		self._audio_buffers[job_id] = io.BytesIO()
		complete = asyncio.Event()
		self._active_generators[job_id] = {
			"generator": gen, 
			"complete_event": complete, 
			"ttfb_done": False,
			"start_time": time.time(),
			"retries": 0
		}

		await gen.push_frame(TTSStartedFrame())
		success = await self._send_generate_request(job_id, text, gen)
		if not success:
			self._cleanup_job(job_id)
			return

		try:
			await asyncio.wait_for(complete.wait(), timeout=self.REQUEST_TIMEOUT)
		except asyncio.TimeoutError:
			# 이 시점에서 타임아웃으로 처리
			await gen.push_frame(ErrorFrame(f"TTS request timed out after {self.REQUEST_TIMEOUT}s"))
			
		# flush 남은 오디오
		buf = self._audio_buffers.pop(job_id, None)
		if buf and buf.tell():
			buf.seek(0)
			await self._push_pcm_chunks(job_id, gen, buf.read())

		await gen.push_frame(TTSStoppedFrame())
		self._cleanup_job(job_id)
		

	async def _send_generate_request(self, job_id: str, text: str, gen: "FrameGenerator") -> bool:
		"""
		서버에 생성 요청을 보내고 필요시 재시도
		"""
		gen_info = self._active_generators.get(job_id)
		if not gen_info:
			return False
			
		max_retries = self.MAX_REQUEST_RETRIES
		current_retry = gen_info["retries"]
		
		while current_retry <= max_retries and job_id in self._active_generators:
			try:
				async with self._safe_cmd_communication() as sock:
					await sock.send_json({
						"command": "generate",
						"job_id": job_id,
						"text": text,
						"voice": self._settings["voice"],
						"speed": float(self._settings["speed"]),
						"target_sample_rate": self.sample_rate,
						"sample_format": "int16",
						"chunk_size": self._chunk_size,
					})
					
					# 응답 대기에 명시적 타임아웃 추가
					try:
						resp = await asyncio.wait_for(sock.recv_json(), timeout=self.REQUEST_TIMEOUT)
					except asyncio.TimeoutError:
						raise RuntimeError(f"Request timed out after {self.REQUEST_TIMEOUT}s")
				
				if resp.get("status") != "started":
					raise RuntimeError(f"TTS failed: {resp}")
				return True
				
			except Exception as e:
				logger.error("Client TTS request failed: {e}")
				
				if current_retry >= max_retries:
					await gen.push_frame(ErrorFrame(f"TTS request failed after {current_retry+1} attempts: {e}"))
					return False
				
				current_retry += 1
				if job_id in self._active_generators:
					self._active_generators[job_id]["retries"] = current_retry
				
				# 재시도 전 잠시 대기
				await asyncio.sleep(self.RECONNECT_DELAY * (current_retry * 0.5))
		return False

	async def _recv_audio(self):
		consecutive_errors = 0
		
		while self._active:
			try:
				try:
					parts = await self._audio_sock.recv_multipart()
					consecutive_errors = 0
				except zmq.error.Again:
					continue
				except Exception as e:
					consecutive_errors += 1
					if consecutive_errors > 3:
						if self._socket_healthy:
							self._socket_healthy = False
						await asyncio.sleep(0.5)
					continue

				if len(parts) != 3:
					continue

				jid, mtype, data = parts
				jid = jid.decode()

				if jid not in self._active_generators:
					self._ignored_msg_counts[jid] = self._ignored_msg_counts.get(jid, 0) + 1
					continue

				gen_info = self._active_generators[jid]
				gen = gen_info["generator"]

				if mtype == b"meta":
					meta = json.loads(data.decode())
					self._remote_sr = meta.get("sample_rate", self.sample_rate)
					self._reset_stats()

				elif mtype == b"data":
					buf = self._audio_buffers.get(jid)
					if not buf:
						continue
						
					buf.write(data)
					self._stats["chunks_received"] += 1
					self._stats["bytes_received"] += len(data)

					# flush 청크
					while buf.tell() >= self._chunk_size:
						buf.seek(0)
						chunk = buf.read(self._chunk_size)
						leftover = buf.read()
						buf.seek(0), buf.truncate(), buf.write(leftover)

						await self._push_pcm_chunks(jid, gen, chunk)
						if not gen_info["ttfb_done"]:
							gen_info["ttfb_done"] = True

				elif mtype == b"end":
					# flush 남은 오디오
					buf = self._audio_buffers.get(jid)
					if buf and buf.tell():
						buf.seek(0)
						await self._push_pcm_chunks(jid, gen, buf.read())
					gen_info["complete_event"].set()

				elif mtype == b"error":
					await gen.push_frame(ErrorFrame(data.decode()))
					gen_info["complete_event"].set()

			except asyncio.CancelledError:
				break
			except Exception as e:
				await asyncio.sleep(0.5)

	async def _push_pcm_chunks(self, jid: str, gen: "FrameGenerator", pcm: bytes):
		try:
			frame = TTSAudioRawFrame(
				audio=pcm,
				sample_rate=self._remote_sr or self.sample_rate,
				num_channels=1,
			)
			await gen.push_frame(frame)
		except Exception as e:
			logger.exception(f"could not push audio chunk: {e}")

	async def _interrupt_job(self, job_id: str):
		"""단일 작업 인터럽트 """
		try:
			async with self._safe_cmd_communication() as sock:
				await sock.send_json({"command": "interrupt", "job_id": job_id})
				await sock.recv_json()
			self._cleanup_job(job_id)
			return True
		except Exception as e:
			# 인터럽트 실패해도 일단 정리
			self._cleanup_job(job_id)
			return False

	async def _interrupt_all_jobs(self):
		"""모든 작업 인터럽트 """
		if not self._active_generators:
			return True
			
		try:
			# 서버로 interrupt 요청
			async with self._safe_cmd_communication() as sock:
				await sock.send_json({"command": "interrupt", "job_id": None})
				await sock.recv_json()
				
			# 모든 작업 정리
			for jid in list(self._active_generators.keys()):
				self._cleanup_job(jid)
			self._job_id = None
			return True
		except Exception as e:
			logger.error(f"failed to interrupt all jobs: {e}")
			# 실패시 강제 완료 처리
			self._force_complete_all_jobs("Interrupt failed")
			return False

	def _cleanup_job(self, jid: str):
		"""단일 작업 정리"""
		self._active_generators.pop(jid, None)
		self._audio_buffers.pop(jid, None)
		self._ignored_msg_counts.pop(jid, None)
		if self._job_id == jid:
			self._job_id = None

	def _force_complete_all_jobs(self, reason: str = "Force completed"):
		"""모든 작업 강제 완료 """
		for jid, info in list(self._active_generators.items()):
			try:
				info["complete_event"].set()
				asyncio.create_task(info["generator"].push_frame(
					ErrorFrame(f"Job interrupted: {reason}")
				))
			except Exception:
				pass
		
		self._active_generators.clear()
		self._audio_buffers.clear()
		self._job_id = None

	async def _socket_watchdog(self):
		"""소켓 모니터링 """
		while self._active:
			try:
				await asyncio.sleep(self.WATCHDOG_INTERVAL)
				
				# 소켓 건강 확인
				if not self._socket_healthy:
					if await self._reset_sockets():
						logger.info("client socket reset successful")
					continue
				
				# 오래된 작업 확인 (REQUEST_TIMEOUT 이상 걸린 작업)
				current_time = time.time()
				stale_jobs = []
				
				for jid, info in self._active_generators.items():
					job_duration = current_time - info["start_time"]
					if job_duration > self.REQUEST_TIMEOUT:
						stale_jobs.append(jid)
						logger.warning(f"[CLIENT] Job {jid} has been running for {job_duration:.1f}s, marking as stale")
						info["complete_event"].set()
						asyncio.create_task(info["generator"].push_frame(
							ErrorFrame(f"Job timed out after {job_duration:.1f}s")
						))
				
				# 마지막 성공 시간 확인
				cmd_idle_time = current_time - self._last_successful_cmd
				if cmd_idle_time > 60.0 and len(self._active_generators) > 0:
					logger.warning(f"[CLIENT] No successful commands for {cmd_idle_time:.1f}s, resetting sockets")
					self._socket_healthy = False
			
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.exception(f"[CLIENT] Watchdog error: {e}")
	
	async def _fetch_voices(self):
		"""음성 목록 가져오기"""
		try:
			async with self._safe_cmd_communication() as sock:
				await sock.send_json({"command": "list_voices"})
				resp = await sock.recv_json()
				
			if resp.get("status") == "success":
				self._available_voices = {v: v for v in resp.get("voices", [])}
				return True
			return False
		except Exception as e:
			logger.error("could not fetch voices: %s", e)
			return False

	def _reset_stats(self):
		self._stats = {"chunks_received": 0, "bytes_received": 0, "start_time": time.time()}

	async def _cleanup_stale_jobs(self):
		"""오래된 작업 정리"""
		while self._active:
			try:
				await asyncio.sleep(60)
				stale = [jid for jid, c in self._ignored_msg_counts.items() if c > 100]
				for jid in stale: 
					self._cleanup_job(jid)
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.exception(f"cleanup task error: {e}")

	async def process_frame(self, frame, direction: str):
		await super().process_frame(frame, direction)
		if isinstance(frame, UserStartedSpeakingFrame):
			await self._interrupt_all_jobs()
			try:
				while not self._request_queue.empty():
					self._request_queue.get_nowait()
					self._request_queue.task_done()
			except Exception as e:
				logger.error(f"cleaning up request queue: {e}")

	async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
		gen = FrameGenerator(self)
		await self._request_queue.put((text, gen))
		async for f in gen: yield f

	async def flush_audio(self): pass   # Pipecat 요구사항용


class FrameGenerator:
	def __init__(self, service):
		self._q = asyncio.Queue()
		self._active = True
		
	async def push_frame(self, f: Frame): 
		if self._active:
			try:
				await self._q.put(f)
			except Exception as e:
				logger.error(f"error pushing frame: {e}")
	
	def __aiter__(self): 
		return self
	
	async def __anext__(self):
		if not self._active and self._q.empty(): 
			raise StopAsyncIteration
		
		try:
			frame = await self._q.get()
			if isinstance(frame, (TTSStoppedFrame, ErrorFrame)):
				self._active = False
			return frame
		except Exception as e:
			logger.exception(f"error getting frame: {e}")
			self._active = False
			raise StopAsyncIteration
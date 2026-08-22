import socket
import struct
import threading

WSJT_X_PORT = 2237
DIREWOLF_PORTS = (8001, 8005)
FLDIGI_PORT = 7362
WSJT_X_MAGIC = 0xadbccbda

def _decode_wsjt_string(data, offset):
    end = data.find(b'\x00', offset)
    if end == -1:
        return data[offset:].decode('utf-8', errors='replace'), len(data) - offset
    return data[offset:end].decode('utf-8', errors='replace'), end - offset + 1

def parse_wsjt_x(data):
    result = {'type': 0, 'raw': data}
    if len(data) < 8:
        return result
    magic, pkt_type = struct.unpack_from('<II', data, 0)
    if magic != WSJT_X_MAGIC:
        return result
    result['type'] = pkt_type
    offset = 8
    if pkt_type == 1:
        if len(data) < offset + 4:
            return result
        id_str, consumed = _decode_wsjt_string(data, offset)
        result['id'] = id_str
        offset += consumed
        if offset + 4 >= len(data):
            return result
        mode_str, consumed = _decode_wsjt_string(data, offset)
        result['mode'] = mode_str
        offset += consumed
        if offset + 4 > len(data):
            return result
        result['frequency'] = struct.unpack_from('<Q', data, offset)[0]
        offset += 8
        if offset + 4 > len(data):
            return result
        result['snr'] = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        if offset + 4 > len(data):
            return result
        result['tx_enabled'] = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        if offset + 4 > len(data):
            return result
        result['transmitting'] = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        if offset + 4 > len(data):
            return result
        result['decoding'] = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        if offset + 4 > len(data):
            return result
        result['dial_frequency'] = struct.unpack_from('<Q', data, offset)[0]
        offset += 8
        if offset + 4 > len(data):
            return result
        result['message_type'] = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        remaining = data[offset:]
        extra = {}
        extra_offset = 0
        while extra_offset < len(remaining):
            try:
                s, c = _decode_wsjt_string(remaining, extra_offset)
                if c > 1:
                    extra[f'field_{extra_offset}'] = s
                extra_offset += c
            except Exception:
                break
        if extra:
            result['extra'] = extra
        return result
    if pkt_type == 2:
        if len(data) < offset + 4:
            return result
        id_str, consumed = _decode_wsjt_string(data, offset)
        result['id'] = id_str
        offset += consumed
        if offset + 4 > len(data):
            return result
        result['time'] = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        if offset + 4 > len(data):
            return result
        result['snr'] = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        if offset + 4 > len(data):
            return result
        result['delta_time'] = struct.unpack_from('<d', data, offset)[0]
        offset += 8
        if offset + 4 > len(data):
            return result
        result['delta_frequency'] = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        if offset + 4 > len(data):
            return result
        message_str, consumed = _decode_wsjt_string(data, offset)
        result['message'] = message_str
        offset += consumed
        if offset + 4 > len(data):
            return result
        decode_str, consumed = _decode_wsjt_string(data, offset)
        result['decode'] = decode_str
        offset += consumed
        remaining = data[offset:]
        extra = {}
        extra_offset = 0
        while extra_offset < len(remaining):
            try:
                s, c = _decode_wsjt_string(remaining, extra_offset)
                if c > 1:
                    extra[f'field_{extra_offset}'] = s
                extra_offset += c
            except Exception:
                break
        if extra:
            result['extra'] = extra
        return result
    if pkt_type == 3:
        if len(data) < offset + 4:
            return result
        id_str, consumed = _decode_wsjt_string(data, offset)
        result['id'] = id_str
        offset += consumed
        result['ack'] = True
        return result
    id_str, consumed = _decode_wsjt_string(data, offset)
    result['id'] = id_str
    return result

def parse_direwolf(data):
    decoded = data.decode('utf-8', errors='replace')
    return {'type': 'aprs', 'line': decoded.strip()}

def parse_fldigi(data):
    decoded = data.decode('utf-8', errors='replace')
    fields = {}
    for part in decoded.split('&'):
        if '=' in part:
            k, v = part.split('=', 1)
            fields[k.strip()] = v.strip().strip("'\"")
    return {'type': 'status', 'data': fields, 'raw': decoded.strip()}

class UdpBroadcastRx:
    def __init__(self, log_fn=None):
        self._log_fn = log_fn
        self._thread = None
        self._running = False
        self._sockets = []
        self._programs = {}
        self._listeners = {}
        self._lock = threading.Lock()
        self._init_defaults()

    def _init_defaults(self):
        self.add_program('wsjt-x', WSJT_X_PORT, parse_wsjt_x)
        self.add_program('direwolf', DIREWOLF_PORTS[0], parse_direwolf)
        self.add_program('direwolf_8005', DIREWOLF_PORTS[1], parse_direwolf)
        self.add_program('fldigi', FLDIGI_PORT, parse_fldigi)

    def _log(self, msg):
        if self._log_fn:
            self._log_fn(msg)

    def add_program(self, name, port, parser):
        with self._lock:
            self._programs[name] = {'port': port, 'parser': parser}
            if name not in self._listeners:
                self._listeners[name] = []
            if name == 'direwolf':
                self._listeners.setdefault('direwolf_8005', [])

    def set_callback(self, program, callback):
        with self._lock:
            if program in self._listeners:
                self._listeners[program].append(callback)
            else:
                self._listeners[program] = [callback]

    def _build_sockets(self):
        for name, info in self._programs.items():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                s.bind(('', info['port']))
                s.settimeout(0.5)
                self._sockets.append((name, s))
                self._log(f"UdpBroadcastRx: listening on port {info['port']} for {name}")
            except Exception as e:
                self._log(f"UdpBroadcastRx: failed to bind port {info['port']} for {name}: {e}")

    def _run(self):
        self._build_sockets()
        if not self._sockets:
            self._log("UdpBroadcastRx: no sockets bound, exiting")
            return
        self._running = True
        while self._running:
            for name, s in self._sockets:
                try:
                    data, _addr = s.recvfrom(65535)
                except TimeoutError:
                    continue
                except OSError:
                    continue
                if not data:
                    continue
                parser = self._programs[name]['parser']
                try:
                    parsed = parser(data)
                except Exception as e:
                    self._log(f"UdpBroadcastRx: parser error for {name}: {e}")
                    continue
                if not parsed:
                    continue
                callbacks = list(self._listeners.get(name, []))
                if name == 'direwolf':
                    callbacks.extend(self._listeners.get('direwolf', []))
                for cb in callbacks:
                    try:
                        cb(name, parsed)
                    except Exception as e:
                        self._log(f"UdpBroadcastRx: callback error for {name}: {e}")

    def start(self):
        with self._lock:
            if self._running:
                self._log("UdpBroadcastRx: already running")
                return
        self._thread = threading.Thread(target=self._run, daemon=True, name='udp-broadcast-rx')
        self._thread.start()
        self._log("UdpBroadcastRx: started")

    def stop(self):
        self._running = False
        for name, s in self._sockets:
            try:
                s.close()
            except OSError:
                pass
        self._sockets.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._log("UdpBroadcastRx: stopped")

    def is_running(self):
        return self._running

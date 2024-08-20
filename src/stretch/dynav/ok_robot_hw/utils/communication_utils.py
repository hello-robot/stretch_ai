import numpy as np
import zmq
import cv2

# use zmq to send a numpy array
def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    A = np.array(A)
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(np.ascontiguousarray(A), flags, copy=copy, track=track)

# use zmq to receive a numpy array
def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])

def send_rgb_img(socket, img):
    img = img.astype(np.uint8) 
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, img_encoded = cv2.imencode('.jpg', img, encode_param)
    socket.send(img_encoded.tobytes())

def recv_rgb_img(socket):
    img = socket.recv()
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img

def send_depth_img(socket, depth_img):
    depth_img = (depth_img * 1000).astype(np.uint16)
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]  # Compression level from 0 (no compression) to 9 (max compression)
    _, depth_img_encoded = cv2.imencode('.png', depth_img, encode_param)
    socket.send(depth_img_encoded.tobytes())

def recv_depth_img(socket):
    depth_img = socket.recv()
    depth_img = np.frombuffer(depth_img, dtype=np.uint8)
    depth_img = cv2.imdecode(depth_img, cv2.IMREAD_UNCHANGED)
    depth_img = (depth_img / 1000.)
    return depth_img
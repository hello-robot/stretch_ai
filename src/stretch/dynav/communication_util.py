import zmq
import numpy as np
import cv2

def load_socket(port_number):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:" + str(port_number))

    return socket

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    A = np.array(A)
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(np.ascontiguousarray(A), flags, copy=copy, track=track)

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

def send_everything(socket, rgb, depth, intrinsics, pose):
    send_rgb_img(socket, rgb)
    socket.recv_string()
    send_depth_img(socket, depth)
    socket.recv_string()
    send_array(socket, intrinsics)
    socket.recv_string()
    send_array(socket, pose)
    socket.recv_string()

def recv_everything(socket):
    rgb = recv_rgb_img(socket)
    socket.send_string('')
    depth = recv_depth_img(socket)
    socket.send_string('')
    intrinsics = recv_array(socket)
    socket.send_string('')
    pose = recv_array(socket)
    socket.send_string('')
    return rgb, depth, intrinsics, pose
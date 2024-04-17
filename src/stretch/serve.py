import multiprocessing
import stretch_body.device


def serve_protocol(port):
    import stretch.comms.send_protocol as sp
    sock, poll = sp.initialize(port)
    while True:
        sp.send_spp(sock, poll)


def serve_body(status_port, moveby_port):
    import stretch.comms.send_body as sb
    status_sock, moveby_sock, moveby_poll, robot = sb.initialize(status_port, moveby_port)
    while True:
        sb.send_status(status_sock, robot)
        sb.exec_moveby(moveby_sock, moveby_poll, robot)
        # sb.send_parameters(sock, robot)
        # sb.send_urdf(sock, robot)


def serve_head_nav_cam(camarr_port, camb64_port):
    import stretch.comms.send_head_nav_cam as shnc
    camarr_sock, camb64_sock, camera = shnc.initialize(camarr_port, camb64_port)
    while True:
        shnc.send_imagery_as_numpy_arr(camarr_sock, camera)
        shnc.send_imagery_as_base64_str(camb64_sock, camera)


def serve_all():
    d = stretch_body.device.Device(name='stretchpy', req_params=False)
    base_port = d.params.get('base_port', 20200)

    # Spawn each component as a separate process
    port = base_port
    serve_protocol_process = multiprocessing.Process(target=serve_protocol, args=(port,))

    port += 1; status_port = port
    port += 1; moveby_port = port
    serve_body_process = multiprocessing.Process(target=serve_body, args=(status_port, moveby_port,))

    port += 1; camarr_port = port
    port += 1; camb64_port = port
    serve_head_nav_cam_process = multiprocessing.Process(target=serve_head_nav_cam, args=(camarr_port, camb64_port,))

    try:
        serve_protocol_process.start()
        serve_body_process.start()
        serve_head_nav_cam_process.start()
        while True:
            pass
    finally:
        serve_protocol_process.terminate()
        serve_protocol_process.join()
        serve_body_process.terminate()
        serve_body_process.join()
        serve_head_nav_cam_process.terminate()
        serve_head_nav_cam_process.join()

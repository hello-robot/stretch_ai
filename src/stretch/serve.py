import multiprocessing
import stretch_body.device


def serve_protocol(port):
    import stretch.comms.send_protocol as sp
    sock, poll = sp.initialize(port)
    while True:
        sp.send_spp(sock, poll)


def serve_body(port):
    import stretch.comms.send_body as sb
    sock, robot = sb.initialize(port)
    while True:
        sb.send_status(sock, robot)
        sb.send_parameters(sock, robot)
        sb.send_urdf(sock, robot)


def serve_all():
    d = stretch_body.device.Device(name='stretchpy', req_params=False)
    base_port = d.params.get('base_port', 20200)

    # Spawn each component as a separate process
    port = base_port
    serve_protocol_process = multiprocessing.Process(target=serve_protocol, args=(port,))

    port += 1
    serve_body_process = multiprocessing.Process(target=serve_body, args=(port,))

    try:
        serve_protocol_process.start()
        serve_body_process.start()
        while True:
            pass
    finally:
        serve_protocol_process.terminate()
        serve_protocol_process.join()
        serve_body_process.terminate()
        serve_body_process.join()

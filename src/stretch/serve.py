import argparse
import multiprocessing
import time
from typing import Optional

import stretch_body.device

import stretch.versions  # defines package-wide version & protocol


def serve_protocol(port):
    import stretch.comms.send_protocol as sp

    sock, poll = sp.initialize(port)
    while True:
        sp.send_spp(sock, poll)


def serve_body(status_port, moveby_port, basevel_port):
    import stretch.comms.send_body as sb

    (
        status_sock,
        moveby_sock,
        moveby_poll,
        basevel_sock,
        basevel_poll,
        body,
    ) = sb.initialize(status_port, moveby_port, basevel_port)
    while True:
        sb.send_status(status_sock, body)
        sb.exec_moveby(moveby_sock, moveby_poll, body)
        sb.exec_basevel(basevel_sock, basevel_poll, body)
        # sb.send_parameters(sock, body)
        # sb.send_urdf(sock, body)


def serve_head_nav_cam(camarr_port, camb64_port):
    import stretch.comms.send_head_nav_cam as shnc

    camarr_sock, camb64_sock, camera = shnc.initialize(camarr_port, camb64_port)
    while True:
        shnc.send_imagery_as_numpy_arr(camarr_sock, camera)
        shnc.send_imagery_as_base64_str(camb64_sock, camera)


def serve_realsense(
    camarr_port, camb64_port, exposure: str = "low", sensor_type="d405"
):
    import stretch.comms.send_realsense as seer

    camarr_sock, camb64_sock, camera = seer.initialize(
        camarr_port, camb64_port, exposure=exposure, sensor_type=sensor_type
    )
    while True:
        msg = seer.send_imagery_as_numpy_arr(camarr_sock, camera)
        # Shrink the message
        seer.send_imagery_as_base64_str(camb64_sock, msg)


def serve_aruco(info_port, add_port, delete_port):
    import stretch.comms.send_aruco as sa

    (
        info_sock,
        info_poll,
        add_sock,
        add_poll,
        delete_sock,
        delete_poll,
        marker_db,
    ) = sa.initialize(info_port, add_port, delete_port)
    while True:
        sa.send_marker_info(info_sock, info_poll, marker_db)
        sa.add_marker(add_sock, add_poll, marker_db)
        sa.delete_marker(delete_sock, delete_poll, marker_db)


class StretchServer:
    def _add_body_process(self, function, port: int) -> int:
        """Add a basic comms process, and let us know which port we are on next."""
        port += 1
        status_port = port
        port += 1
        moveby_port = port
        port += 1
        basevel_port = port
        process = multiprocessing.Process(
            target=function,
            args=(
                status_port,
                moveby_port,
                basevel_port,
            ),
        )
        self._processes.append(process)
        return port

    def _add_cam_process(self, function, port: int) -> int:
        """Add a basic comms process, and let us know which port we are on next."""
        port += 1
        camarr_port = port
        port += 1
        camb64_port = port
        process = multiprocessing.Process(
            target=function,
            args=(
                camarr_port,
                camb64_port,
            ),
        )
        self._processes.append(process)
        return port

    def _add_aruco_process(self, function, port: int) -> int:
        port += 1
        info_port = port
        port += 1
        add_port = port
        port += 1
        delete_port = port
        process = multiprocessing.Process(
            target=function,
            args=(
                info_port,
                add_port,
                delete_port,
            ),
        )
        self._processes.append(process)
        return port

    def __init__(
        self,
        port_offset: Optional[int] = None,
        ee_exposure: str = "low",
        head_exposure: str = "auto",
    ):
        """Create the processes that need to be created"""
        port_offset = 20200 if port_offset is None else port_offset
        self._processes = []

        self.device = stretch_body.device.Device(name="stretchpy", req_params=False)
        base_port = self.device.params.get("base_port", port_offset)

        # Spawn each component as a separate process
        port = base_port
        serve_protocol_process = multiprocessing.Process(
            target=serve_protocol, args=(port,)
        )
        self._processes.append(serve_protocol_process)

        # Spawn each component as a separate process
        port = self._add_body_process(serve_body, port)
        port = self._add_cam_process(serve_head_nav_cam, port)

        def serve_ee_realsense(port1, port2):
            return serve_realsense(
                port1, port2, exposure=ee_exposure, sensor_type="d405"
            )

        port = self._add_cam_process(serve_ee_realsense, port)

        def serve_head_realsense(port1, port2):
            return serve_realsense(
                port1, port2, exposure=head_exposure, sensor_type="d435i"
            )

        port = self._add_cam_process(serve_head_realsense, port)
        port = self._add_aruco_process(serve_aruco, port)
        print("Server is done adding procs.")

    def spin(self):
        """Spin for a while, letting all the threads keep serving."""
        print("Server is running!")
        try:
            for process in self._processes:
                process.start()
            while True:
                time.sleep(1e-5)
        finally:
            for process in self._processes:
                process.terminate()
                process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StretchPy robot server")
    parser.add_argument(
        "--port", type=int, help="Set the port offset for StretchPy's sockets"
    )
    args, _ = parser.parse_known_args()

    print(f"StretchPy Server v{stretch.versions.__version__}")
    server = StretchServer(args.port)
    server.spin()

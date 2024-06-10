from stretch.comms import recv_aruco, recv_protocol
from stretch.utils import auth


class MarkersDatabase:
    def __init__(self, client=None):
        if client is not None:
            ip_addr = client.ip_addr
            port = client.port
        else:
            ip_addr, port = auth.get_robot_address(0)

        # Verify protocol
        proto_port = port
        proto_sock = recv_protocol.initialize(ip_addr, proto_port)
        recv_protocol.recv_spp(proto_sock)

        # Connect to aruco database
        info_port = port + 10
        add_port = port + 11
        delete_port = port + 12
        self.info_sock, self.add_sock, self.delete_sock = recv_aruco.initialize(
            ip_addr, info_port, add_port, delete_port
        )

    def get_markers(self):
        return recv_aruco.recv_marker_info(self.info_sock)

    def add_new_marker(self,
        marker_id : int,
        marker_name : str,
        length_mm : float,
        link_name : str,
        use_rgb_only : bool = False
    ):
        new_marker = {
            str(int(marker_id)): {
                'length_mm': float(length_mm),
                'use_rgb_only': bool(use_rgb_only),
                'name': str(marker_name),
                'link': str(link_name),
            }
        }
        recv_aruco.send_add_request(self.add_sock, new_marker)

    def delete_marker(self, marker_id : int):
        marker_id = str(int(marker_id))
        recv_aruco.send_delete_request(self.delete_sock, marker_id)

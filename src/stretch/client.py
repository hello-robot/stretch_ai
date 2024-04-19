import numbers

from stretch.comms import recv_body, recv_head_nav_cam, recv_protocol
from stretch.exceptions.connection import NotConnectedException
from stretch.exceptions.motion import MoveByMotionNotAcceptedException
from stretch.utils import auth


class StretchClient:
    """This class creates a client connection to a Stretch robot."""

    def __init__(self, index: int = 0, ip_addr: str = None, port: int = None):
        """Initializes client to first robot by default

        Args:
          index: Defines which robot to connect to
          ip_addr: Instead of index, connect to a robot at "ip_addr:port"
          port: Instead of index, connect to a robot at "ip_addr:port"
        """
        # Get address
        if not (ip_addr and port):
            ip_addr, port = auth.get_robot_address(index)
        self.ip_addr = ip_addr
        self.port = port

    def connect(self):
        """Connects client to robot"""
        # Verify protocol
        proto_port = self.port
        proto_sock = recv_protocol.initialize(self.ip_addr, proto_port)
        recv_protocol.recv_spp(proto_sock)

        # Connect to body
        status_port = self.port + 1
        moveby_port = self.port + 2
        self.status_sock, self.moveby_sock = recv_body.initialize(
            self.ip_addr, status_port, moveby_port
        )

        # Connect to head nav camera
        hncarr_port = self.port + 3
        hncb64_port = self.port + 4
        _, self.hncb64_sock = recv_head_nav_cam.initialize(
            self.ip_addr, hncarr_port, hncb64_port
        )

        self.connected = True

    def disconnect(self):
        """Disconnects client from robot"""
        self.connected = False

    def get_status(self) -> dict:
        """Returns joint state"""
        if not self.connected:
            raise NotConnectedException("use the connect() method")
        return recv_body.recv_status(self.status_sock)

    def move_by(
        self,
        *args,
        joint_translate: float = None,
        joint_rotate: float = None,
        joint_lift: float = None,
        joint_arm: float = None,
        joint_wrist_yaw: float = None,
        joint_wrist_pitch: float = None,
        joint_wrist_roll: float = None,
        joint_gripper: float = None,
        joint_head_pan: float = None,
        joint_head_tilt: float = None,
        **kwargs,
    ):
        """Moves robot's joints by a given amount.

        Only keyword arguments are accepted (e.g. joint_arm=0.1).
        Prismatic joints are in meters, and revolute joints are in
        radians. Cannot use joint_translate & joint_rotate of mobile
        base simultaneously. Raises exception on invalid commands.
        """
        if not self.connected:
            raise NotConnectedException("use the connect() method")
        if args:
            raise ValueError("This method only accepted keyword arguments")
        kwargs["joint_translate"] = joint_translate
        kwargs["joint_rotate"] = joint_rotate
        kwargs["joint_lift"] = joint_lift
        kwargs["joint_arm"] = joint_arm
        kwargs["joint_wrist_yaw"] = joint_wrist_yaw
        kwargs["joint_wrist_pitch"] = joint_wrist_pitch
        kwargs["joint_wrist_roll"] = joint_wrist_roll
        kwargs["joint_gripper"] = joint_gripper
        kwargs["joint_head_pan"] = joint_head_pan
        kwargs["joint_head_tilt"] = joint_head_tilt
        pose = {joint: kwargs[joint] for joint in kwargs if kwargs[joint] is not None}
        for joint, moveby_amount in pose.items():
            if not isinstance(moveby_amount, numbers.Real):
                raise MoveByMotionNotAcceptedException(
                    f"Cannot move {joint} by {moveby_amount} amount"
                )
        if pose:
            recv_body.send_moveby(self.moveby_sock, pose)

    def take_nav_picture(self):
        """Returns numpy BGR image from Head Nav camera"""
        if not self.connected:
            raise NotConnectedException("use the connect() method")
        return recv_head_nav_cam.recv_imagery_as_base64_str(self.hncb64_sock)

    def stream_nav_camera(self):
        """Returns Python generator to stream numpy BGR imagery
        from Head Nav camera"""
        if not self.connected:
            raise NotConnectedException("use the connect() method")
        while True:
            yield self.take_nav_picture()

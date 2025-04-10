import time
import threading
from concurrent import futures
import smpc_pb2
import smpc_pb2_grpc
import grpc
import structlog
import numpy as np
logger = structlog.get_logger()

class SMPCServicer(smpc_pb2_grpc.SMPCServicer):
    def __init__(self, client):
        self.client = client

    def SendShares(self, request, context):
        client_id = request.client_id
        shares = []
        for share in request.shares:
            shape = tuple(share.shape)
            data = np.frombuffer(share.data, dtype=np.float32)
            if data.size != np.prod(shape):
                logger.error(f"Mismatch in data size and shape: {data.size} != {np.prod(shape)}")
                return smpc_pb2.AckResponse(status="ERROR")
            shares.append(data.reshape(shape))
        self.client.receive_shares(client_id, shares)
        return smpc_pb2.AckResponse(status="ACK")

class SMPCClient():
    def __init__(self, client_id, port):
        self.client_id = client_id
        self.peer_addresses = []
        self.peer_stubs = {}
        self.all_peers_connected = threading.Event()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        smpc_pb2_grpc.add_SMPCServicer_to_server(SMPCServicer(self), self.server)
        self.server.add_insecure_port(f"[::]:{port}")
        self.server.start()

    def connect_to_peers(self, peer_addresses, max_retries=5, retry_delay=2):
        """
        Attempts to connect to all peers, retrying if necessary until all are connected.
        """
        time.sleep(2)  # Allow time for server to start
        self.peer_addresses = peer_addresses
        logger.info(f"Client {self.client_id} connecting to peers")

        for attempt in range(max_retries):
            self.peer_stubs.clear()

            for i, peer_address in enumerate(self.peer_addresses):
                try:
                    peer_id = i  # Temporary ID for addressing
                    channel = grpc.insecure_channel(peer_address)
                    self.peer_stubs[peer_id] = smpc_pb2_grpc.SMPCStub(channel)
                    logger.info(f"Client {self.client_id} connected to peer at {peer_address}")
                except Exception as e:
                    logger.error(f"Client {self.client_id} failed to connect to peer at {peer_address}: {e}")

            if len(self.peer_stubs) == len(self.peer_addresses):
                logger.info(f"Client {self.client_id} connected to all peers")
                self.all_peers_connected.set()
                return self.peer_stubs

            logger.warning(f"Client {self.client_id} failed to connect to all peers. Retrying ({attempt + 1}/{max_retries})...")
            time.sleep(retry_delay)

        logger.error(f"Client {self.client_id} could not establish connections to all peers after {max_retries} attempts.")
        return self.peer_stubs

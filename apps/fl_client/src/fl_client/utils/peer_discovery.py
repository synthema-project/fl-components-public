import logging
import threading
import time
import socket
import json
import uuid
import os
import requests
from typing import Dict, List, Optional

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants for discovery
DISCOVERY_PORT = 9876
DISCOVERY_BROADCAST_ADDR = '<broadcast>'
DISCOVERY_INTERVAL = 3  # seconds
DISCOVERY_TIMEOUT = 30  # seconds

# Kubernetes related constants
KUBERNETES_SERVICE_HOST = os.environ.get('KUBERNETES_SERVICE_HOST')
IS_KUBERNETES = KUBERNETES_SERVICE_HOST is not None
K8S_NAMESPACE = os.environ.get('POD_NAMESPACE', 'default')
K8S_APP_LABEL = os.environ.get('APP_LABEL', 'fl-client')
K8S_API_URL = f"https://{KUBERNETES_SERVICE_HOST}:443" if IS_KUBERNETES else ""

class PeerDiscovery:
    """
    Module for peer discovery in a decentralized network.

    This module provides the `PeerDiscovery` class, which enables clients to discover 
    and communicate with peers using UDP broadcast announcements. The discovery 
    process consists of broadcasting the client's presence and listening for other 
    peer announcements within a specified time limit.

    Features:
    - Broadcasts presence using UDP to announce availability.
    - Listens for peer announcements and maintains a dictionary of discovered peers.
    - Supports a minimum peer requirement before completing the discovery process.
    - Allows querying discovered peers and their addresses.
    - Implements a shutdown mechanism to stop discovery.
    """

    def __init__(self, client_id, grpc_port, min_peers=2, max_discovery_time=60):
        self.client_id = client_id
        self.grpc_port = grpc_port
        self.min_peers = min_peers
        self.max_discovery_time = max_discovery_time
        self.uuid = str(uuid.uuid4())  # Unique identifier for this client

        # Dictionary to store discovered peers {client_id: ip_address:port}
        self.peers = {}
        self.discovery_complete = threading.Event()
        self.shutdown_flag = threading.Event()

        # Get local IP
        self.local_ip = self._get_local_ip()
        logger.info(f"Local IP: {self.local_ip}")
        
        # Check if running in Kubernetes
        self.is_kubernetes = IS_KUBERNETES
        if self.is_kubernetes:
            logger.info("Running in Kubernetes environment")

    def _get_local_ip(self):
        """Get the local non-loopback IP address"""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Doesn't need to be reachable
            s.connect(('10.255.255.255', 1))
            local_ip = s.getsockname()[0]
        except Exception:
            local_ip = '127.0.0.1'
        finally:
            s.close()
        return local_ip

    def start_discovery(self):
        """Start the discovery process (broadcast and listen)"""
        # Choose discovery method based on environment
        if self.is_kubernetes:
            threading.Thread(target=self._k8s_discover_peers, daemon=True).start()
        else:
            # Traditional UDP broadcast for non-K8s environments
            threading.Thread(target=self._broadcast_presence, daemon=True).start()
            threading.Thread(target=self._listen_for_peers, daemon=True).start()

        # Wait for discovery to complete or timeout
        start_time = time.time()
        while (len(self.peers) < self.min_peers and
               time.time() - start_time < self.max_discovery_time and
               not self.shutdown_flag.is_set()):
            time.sleep(1)

        if len(self.peers) >= self.min_peers:
            logger.info(f"Discovery complete, found {len(self.peers)} peers: {self.peers}")
            self.discovery_complete.set()
            return True
        else:
            logger.warning(f"Discovery timeout, only found {len(self.peers)} peers")
            self.discovery_complete.set()
            return False

    def _broadcast_presence(self):
        """Broadcast presence to the network"""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

            # Prepare the announcement message
            announce_msg = {
                "type": "announce",
                "client_id": self.client_id,
                "grpc_port": self.grpc_port,
                "uuid": self.uuid,
                "ip": self.local_ip
            }

            # Broadcast until discovery is complete
            while not self.discovery_complete.is_set() and not self.shutdown_flag.is_set():
                try:
                    s.sendto(json.dumps(announce_msg).encode(),
                             (DISCOVERY_BROADCAST_ADDR, DISCOVERY_PORT))
                    logger.debug(f"Broadcasted presence: {announce_msg}")
                except Exception as e:
                    logger.error(f"Error broadcasting presence: {e}")

                time.sleep(DISCOVERY_INTERVAL)

    def _listen_for_peers(self):
        """Listen for peer announcements"""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', DISCOVERY_PORT))
            s.settimeout(1)  # Short timeout to check shutdown flag

            # Listen until discovery is complete
            while not self.discovery_complete.is_set() and not self.shutdown_flag.is_set():
                try:
                    data, addr = s.recvfrom(1024)
                    msg = json.loads(data.decode())

                    # Skip our own broadcasts and unknown message types
                    if msg.get("uuid") == self.uuid or msg.get("type") != "announce":
                        continue

                    # Add peer to the list
                    client_id = msg.get("client_id")
                    ip = msg.get("ip", addr[0])  # Use provided IP or fallback to sender's IP
                    grpc_port = msg.get("grpc_port")

                    if client_id is not None and grpc_port is not None:
                        self.peers[client_id] = f"{ip}:{grpc_port}"
                        logger.info(f"Discovered peer {client_id} at {ip}:{grpc_port}")

                        # Reply to announce our presence too
                        self._send_direct_announcement(addr[0], DISCOVERY_PORT)

                except socket.timeout:
                    pass
                except Exception as e:
                    logger.error(f"Error listening for peers: {e}")

    def _send_direct_announcement(self, ip, port):
        """Send a direct announcement to a specific peer"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                announce_msg = {
                    "type": "announce",
                    "client_id": self.client_id,
                    "grpc_port": self.grpc_port,
                    "uuid": self.uuid,
                    "ip": self.local_ip
                }
                s.sendto(json.dumps(announce_msg).encode(), (ip, port))
        except Exception as e:
            logger.error(f"Error sending direct announcement: {e}")

    def get_peer_addresses(self):
        """Return the discovered peer addresses"""
        self.discovery_complete.wait()
        return list(self.peers.values())

    def get_peer_ids(self):
        """Return the discovered peer IDs"""
        self.discovery_complete.wait()
        return list(self.peers.keys())

    def get_peer_count(self):
        """Return the number of discovered peers"""
        return len(self.peers)

    def _k8s_discover_peers(self):
        """Discover peers using Kubernetes API"""
        token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
        ca_cert_path = '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt'
        
        if not os.path.exists(token_path) or not os.path.exists(ca_cert_path):
            logger.error("Kubernetes service account token or CA cert not found")
            return
            
        with open(token_path, 'r') as f:
            token = f.read().strip()
            
        headers = {
            'Authorization': f'Bearer {token}'
        }
        
        # Keep trying to discover peers until we have enough or timeout
        while (len(self.peers) < self.min_peers and
               not self.discovery_complete.is_set() and
               not self.shutdown_flag.is_set()):
            try:
                # Get all pods with the same app label in the namespace
                url = f"{K8S_API_URL}/api/v1/namespaces/{K8S_NAMESPACE}/pods"
                response = requests.get(
                    url, 
                    headers=headers, 
                    params={'labelSelector': f'app={K8S_APP_LABEL}'},
                    verify=ca_cert_path
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to get pods: {response.status_code} {response.text}")
                    time.sleep(DISCOVERY_INTERVAL)
                    continue
                    
                pods = response.json().get('items', [])
                
                # Filter out our own pod and pods that aren't running
                my_hostname = socket.gethostname()
                for pod in pods:
                    pod_name = pod.get('metadata', {}).get('name', '')
                    pod_status = pod.get('status', {}).get('phase', '')
                    
                    # Skip our own pod and non-running pods
                    if pod_name == my_hostname or pod_status != 'Running':
                        continue
                        
                    # Get pod IP and add to peers
                    pod_ip = pod.get('status', {}).get('podIP')
                    if pod_ip:
                        # Use pod name as client_id for now
                        self.peers[pod_name] = f"{pod_ip}:{self.grpc_port}"
                        logger.info(f"Discovered K8s peer {pod_name} at {pod_ip}:{self.grpc_port}")
                        
            except Exception as e:
                logger.error(f"Error discovering K8s peers: {e}")
                
            time.sleep(DISCOVERY_INTERVAL)
            
    def _k8s_headless_service_discovery(self):
        """Alternative discovery using K8s headless service DNS"""
        # This is a simpler approach if you have a headless service set up
        service_name = os.environ.get('HEADLESS_SERVICE_NAME', 'fl-client-headless')
        
        try:
            # Try to resolve all IPs for the headless service
            ips = socket.gethostbyname_ex(f"{service_name}.{K8S_NAMESPACE}.svc.cluster.local")
            logger.info(f"Resolved headless service to: {ips}")
            
            # Add all IPs except our own
            for ip in ips[2]:  # ips[2] contains the list of IPs
                if ip != self.local_ip:
                    # Generate a client ID from IP
                    client_id = f"client-{ip.replace('.', '-')}"
                    self.peers[client_id] = f"{ip}:{self.grpc_port}"
                    logger.info(f"Discovered K8s peer via DNS: {client_id} at {ip}:{self.grpc_port}")
        except Exception as e:
            logger.error(f"Error in headless service discovery: {e}")

    def shutdown(self):
        """Shutdown the discovery process"""
        self.shutdown_flag.set()

"""Private controller: authenticated HTTP to the restricted SSH GPU gateway."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import threading
import socketserver
import os

GATEWAY_TIMEOUT = 90


def make_handler(config, fixed_client=None):
    clients = config.get('clients', {})
    if fixed_client is not None:
        clients = {'socket-role': fixed_client}
    locks = {scope: threading.Lock() for scope in {item['scope'] for item in clients.values()}}

    def gateway(value):
        command = ['ssh','-i',config['ssh_key_file'],'-o','BatchMode=yes',
            '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+config['known_hosts_file'],
            '-o','ConnectTimeout=10','-p',str(config['ssh_port']),config['ssh_target'],
            'dd-restricted-json-gateway']
        result = subprocess.run(command,input=json.dumps(value,allow_nan=False).encode(),
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=GATEWAY_TIMEOUT)
        try:
            response=json.loads(result.stdout)
        except ValueError:
            raise RuntimeError('GPU gateway transport failed')
        if not isinstance(response,dict):raise RuntimeError('Invalid GPU gateway response')
        return response, result.returncode

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(20)

        def log_message(self, format, *args):
            # Authentication values, query strings and submitted source are
            # never put into an HTTP access log.
            return

        def respond(self, status, value):
            raw=json.dumps(value,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)))
            self.end_headers();self.wfile.write(raw)

        def client(self):
            if fixed_client is not None:
                return fixed_client
            provided=self.headers.get('Authorization','').removeprefix('Bearer ')
            value=clients.get(provided)
            if not value:raise PermissionError('Unknown controller authorization')
            return value

        def do_GET(self):
            try:
                client=self.client()
                if self.path=='/health':
                    response,code=gateway({'operation':'health'})
                elif re.fullmatch(r'/jobs/[a-f0-9]{32}',self.path):
                    response,code=gateway({'operation':'status','scope':client['scope'],
                                          'job_id':self.path.rsplit('/',1)[-1]})
                else:
                    self.respond(404,{'status':'REJECTED'});return
                self.respond(200 if code==0 else 422,response)
            except PermissionError:
                self.respond(401,{'status':'REJECTED'})
            except Exception:
                self.respond(503,{'status':'UNAVAILABLE'})

        def do_POST(self):
            try:
                client=self.client()
                if self.path!='/jobs':
                    self.respond(404,{'status':'REJECTED'});return
                size=int(self.headers.get('Content-Length','0'))
                if not 1<=size<=262144:raise ValueError('Invalid request size')
                payload=self.rfile.read(size)
                if len(payload)!=size:raise ValueError('Truncated request')
                value=json.loads(payload)
                if not isinstance(value,dict) or set(value)!={'source'}:
                    raise ValueError('Only method source can be submitted')
                # The HTTP client cannot choose another scope or change the
                # trusted calibration/formal mode associated with its identity.
                with locks[client['scope']]:
                    response,code=gateway({'operation':'submit','scope':client['scope'],
                                          'source':value['source'],'mode':client['mode']})
                self.respond(202 if code==0 else 422,response)
            except PermissionError:
                self.respond(401,{'status':'REJECTED'})
            except (ValueError,TypeError):
                self.respond(422,{'status':'REJECTED'})
            except Exception:
                self.respond(503,{'status':'UNAVAILABLE'})

    return Handler


class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self.request_slots = threading.BoundedSemaphore(2)
        super().__init__(*args, **kwargs)

    def process_request(self, request, address):
        # A track cannot create unbounded controller threads or SSH processes
        # by flooding health/status requests while another track is running.
        if not self.request_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.request_slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.request_slots.release()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    servers=[]
    for listener in config.get('socket_listeners', []):
        path=Path(listener['socket_file'])
        path.parent.mkdir(parents=True,exist_ok=True)
        # Never unlink an active service's endpoint on an ambiguous restart.
        server=UnixHTTPServer(str(path),make_handler(config, {
            'scope':listener['scope'],'mode':listener['mode']}))
        os.chmod(path,0o666)
        servers.append(server)
    if 'listen_port' in config:
        servers.append(ThreadingHTTPServer(
            (config.get('listen_host','127.0.0.1'),config['listen_port']),make_handler(config)))
    if not servers:raise ValueError('No controller listeners configured')
    for server in servers[:-1]:
        threading.Thread(target=server.serve_forever,daemon=True).start()
    print(json.dumps({'status':'READY','controller':'restricted-dd-http',
                      'listeners':len(servers)}),flush=True)
    try:servers[-1].serve_forever()
    finally:
        for server in servers:server.server_close()


if __name__=='__main__':main()

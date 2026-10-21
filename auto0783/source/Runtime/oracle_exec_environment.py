"""Author-only /solution exec mount; ordinary Agent/Verifier code is unchanged."""
import json
from hybrid_harbor_environment import HybridDockerEnvironment


class OracleExecDockerEnvironment(HybridDockerEnvironment):
    def __init__(self, *args, **kwargs):
        if kwargs.get('author_oracle') is not True:
            raise ValueError('This extension is only for an explicit native author Oracle')
        super().__init__(*args, **kwargs)
        # Docker tmpfs defaults to noexec. Native Harbor Oracle directly executes
        # solve.sh after chmod, so its exclusive author mount needs explicit exec.
        # No formal candidate role receives /solution or this extension.
        if self.author_oracle:
            compose = json.loads(self.compose_path.read_text())
            mounts = compose['services']['main']['tmpfs']
            before = '/solution:rw,nosuid,nodev,size=4m,uid=1000,gid=1000,mode=1777'
            if mounts.count(before) != 1:
                raise RuntimeError('Author mount differs from the diagnosed native Oracle mount')
            mounts[mounts.index(before)] = before.replace('rw,nosuid', 'rw,exec,nosuid')
            self.compose_path.write_text(json.dumps(compose, indent=2) + '\n')

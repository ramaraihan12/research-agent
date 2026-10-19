# Which TCP port does PostgreSQL listen on by default, and how do you change it?

PostgreSQL listens on TCP port 5432 by default for all IP addresses it binds to [1][2]. This default applies even if the `listen_addresses` setting is restricted to `localhost`, which limits connections to the local machine [1]. To change this port, you must modify the `postgresql.conf` configuration file and restart the server service [2].

*   **Configuration**: Locate the `postgresql.conf` file and find the line `#port = 5432`. Uncomment this line and assign a new value between 1024 and 65535, such as `port = 5433` [2].
*   **Restart Requirement**: The new port setting only takes effect after the PostgreSQL service is restarted. On Linux/Unix, this is typically done via `sudo systemctl restart postgresql`, while Windows users must restart the service through `services.msc` [2].
*   **Verification**: You can confirm the change by checking active ports with `sudo netstat -plnt | grep postgres` or by connecting to the new port using the `psql` client with the `-p` flag (e.g., `psql -p 5433`) [2].
*   **Constraints**: The `port` parameter cannot be changed dynamically; it must be set at server start [1]. Additionally, if `listen_addresses` is left empty, the server will not listen on any TCP/IP interface, relying instead on Unix-domain sockets [1].

These sources do not fully settle on the specific security implications of changing the port or the exact behavior of port conflicts when multiple PostgreSQL instances run on the same machine without distinct configuration management.

## Sources

1. [PostgreSQL: Documentation: 18: 19.3. Connections and ...](https://www.postgresql.org/docs/current/runtime-config-connection.html)
2. [How to Change the Default Port in PostgreSQL - GeeksforGeeks](https://www.geeksforgeeks.org/postgresql/how-to-change-the-default-port-in-postgresql/)

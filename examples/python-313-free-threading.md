# Is the GIL still enabled by default in Python 3.13, and how do you try the free-threaded build?

Yes, the Global Interpreter Lock (GIL) remains enabled by default in Python 3.13, as the standard CPython implementation still relies on it to limit parallelism within a single process [1]. However, Python 3.13 introduced an experimental free-threaded build that disables the GIL, allowing true parallel execution of bytecode across multiple CPU cores [1][2].

To attempt the free-threaded build, users must compile Python from source using the `--disable-gil` configure option, as official installers for this variant were not universally available until later updates [2][4]. Once built, the interpreter can be verified by checking if `sysconfig.get_config_var("Py_GIL_DISABLED")` returns 1 or if `python -VV` displays "free-threading build" [2][4]. If a free-threaded build is unavailable, the GIL can be explicitly disabled at runtime by setting the `PYTHON_GIL=0` environment variable, provided the build supports this feature [4].

The sources do not fully settle on the long-term stability of this feature in Python 3.13 specifically, as some documentation notes it was experimental in 3.13 but became non-experimental in 3.14, creating ambiguity regarding the exact status of the 3.13 release cycle [1][3].

## Sources

1. [Global interpreter lock - Wikipedia](https://en.wikipedia.org/wiki/Global_Interpreter_Lock)
2. [Python support for free threading — Python 3.15.0 documentation](https://docs.python.org/3/howto/free-threading-python.html)
3. [Python Free-Threading Guide](https://py-free-threading.github.io/)
4. [Migrating Python Codebases to Free-Threaded CPython 3.13](https://www.locionic.com/en/blog/python-free-threaded-cpython-migration)

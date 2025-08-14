Unlike pull-based decoder (also known as "sans-io"), where the IO is fetched on demand, push-based decode fetches IO in advance and send the IO to the compute sink.

Prior/concurrent art: 
- https://github.com/apache/arrow-rs/issues/8000
- https://github.com/apache/arrow-rs/issues/7983


Sans-io:
- https://fasterthanli.me/articles/the-case-for-sans-io
- https://sans-io.readthedocs.io



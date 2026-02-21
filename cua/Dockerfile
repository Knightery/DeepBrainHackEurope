FROM ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest

USER root
COPY data_fetcher.py /home/computeruse/data_fetcher.py
RUN chown computeruse:computeruse /home/computeruse/data_fetcher.py
USER computeruse

ENTRYPOINT ["/bin/bash", "-c", "./start_all.sh && ./novnc_startup.sh && echo '=== CUA ready === VNC :5900 | noVNC http://localhost:6080' && python /home/computeruse/data_fetcher.py \"$@\"", "--"]

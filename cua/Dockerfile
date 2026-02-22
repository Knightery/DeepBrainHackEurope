FROM ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest

USER root
COPY data_fetcher.py /home/computeruse/data_fetcher.py
RUN chown computeruse:computeruse /home/computeruse/data_fetcher.py && \
    pip install --no-cache-dir python-dotenv

# Reduce bash tool timeout from 120s to 30s
RUN sed -i 's/_timeout: float = 120\.0/_timeout: float = 30.0/g' \
        /home/computeruse/computer_use_demo/tools/bash.py && \
    sed -i 's/timeout: float | None = 120\.0/timeout: float | None = 30.0/g' \
        /home/computeruse/computer_use_demo/tools/run.py

USER computeruse

ENTRYPOINT ["/bin/bash", "-c", "./start_all.sh && ./novnc_startup.sh && echo '=== CUA ready === VNC :5900 | noVNC http://localhost:6080' && python /home/computeruse/data_fetcher.py \"$@\"", "--"]

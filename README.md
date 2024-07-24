# NPI


## Docker instructions

Here, <container-name> can be anything and <docker-image> is pytorch_npi

docker create --name <container-name> -it --shm-size=20G -v <src>:<tgt> --gpus all <docker-image> /bin/bash

Volume lines: Base folder in NAS is /mnt/nas/danielp
    -v /mnt/nas/danielp/repos:/home/daniel/repos
    -v /mnt/nas/danielp/npi:/home/daniel/npi
    -v /home/danielp/npi-remote:/home/daniel/local


Create a screen session:
    screen -S <session-name>
    Detach: Ctrl + A + D
    Reattach: screen -r <session-name>
    Quit: screen -X -S <sesion-name> quit

In screen session:
    docker start <container-name>
    docker exec -it <container-name> /bin/bash
    cd /home/daniel

Run training and exporting scripts within session

Stop Docker session: docker stop <container-name>
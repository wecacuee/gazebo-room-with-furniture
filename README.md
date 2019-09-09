# Gazebo room with furniture

1. Install gazebo-9, python3-pip etc.
    ```

3. Run to download and wrap ikea meshes to models

    ``` shellsession
    source setup.bash
    rm models/ikea_models/meshes
    pip install -r models/ikea_models/pip-requirements.txt
    cd models && python ikea_models/create_models.py
    ```

4. Checkout the worlds

    ``` shellsession
    source setup.bash
    gazebo models/ikea_models/ikea.world
    gazebo ./AtkHall6thFloorWithFurniture.world`
    ```
    
6. Create more worlds

  ``` shellsession
  pip install -r ./worlds/pip-requirements.txt
  python ./worlds/random_world.py
  ```

    

## Generate random worlds

1. Download and generate ikea models.  [[Follow instructions
here.][../models/ikea_models/README.md]]

``` shellsession
source ../setup.bash
```

``` shellsession
pip install -r requirements.txt
python random_world.py --dest_file_fmt
$IG_LEARNING_DATA_DIR/worlds/world%d.sdf
```


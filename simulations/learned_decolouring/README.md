Hi Tom,

The "prepare_flow_phantom_data" scripts can be used to transform the flow phantom in your folder strutures into a npz file.
This npz file can then be loaded and analysed with the "flow_phantom_data_visualisation" script.

Per default this is set to one of the folders on the I: drive.
Feel free to add the other data sets that you have converted :)

You can change the used training data in line 23: TRAINING_DATA = ["BASE", "WATER_4cm", "ILLUM_POINT"]

The legal data set names are:

|Dataset|
|---|
|ACOUS|
|BASE|
|BG_0-100|
|BG_60-80|
|BG_H2O|
|HET_0-100|
|HET_60-80|
|HET_BG|
|ILLUM_5mm|
|ILLUM_POINT|
|INVIS|
|INVIS_ACOUS|
|INVIS_SKIN|
|INVIS_ACOUS_SKIN|
|MSOT|
|MSOT_ACOUS|
|MSOT_SKIN|
|MSOT_ACOUS_SKIN|
|RES_0.6|
|RES_0.15|
|RES_0.15_SMALL|
|RES_1.2|
|SKIN|
|SMALL|
|WATER_2cm|
|WATER_4cm|

in case there is a typo, these names are also part of the GBR filename.

Let me know if there are any questions, please!

Cheers,
Janek
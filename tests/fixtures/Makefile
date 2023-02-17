extra: extra-cog extra-l1c

extra-cog:
	aws s3 cp --no-sign-request s3://sentinel-cogs/sentinel-s2-l2a-cogs/35/U/NV/2018/9/S2A_35UNV_20180921_0_L2A/B04.tif ./extra/B04.tif
	aws s3 cp --no-sign-request s3://sentinel-cogs/sentinel-s2-l2a-cogs/35/U/NV/2018/9/S2A_35UNV_20180921_0_L2A/B09.tif ./extra/B09.tif

extra-l1c:
	aws s3 cp --request-payer requester s3://sentinel-s2-l1c/tiles/35/U/NV/2018/9/21/0/B02.jp2 ./extra/B02_10m.jp2
	aws s3 cp --request-payer requester s3://sentinel-s2-l1c/tiles/35/U/NV/2018/9/21/0/B05.jp2 ./extra/B05_20m.jp2
	aws s3 cp --request-payer requester s3://sentinel-s2-l1c/tiles/35/U/NV/2018/9/21/0/B09.jp2 ./extra/B09_60m.jp2

	aws s3 cp --request-payer requester s3://sentinel-s2-l1c/tiles/38/T/LR/2017/5/18/0/B08.jp2 ./extra/S2A_MSIL1C_T38TLR_20170518_B08_bad.jp2

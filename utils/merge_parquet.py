import os
import pyarrow.parquet as pq

PARAS = ["0.50_0.50_2.00"]
SIZE = 1000

# PARQUETS = ["replace_blend_reweight_foggy_%s_%s_cn_filtered" % (PARA, SIZE), "replace_blend_reweight_night_%s_%s_cn_filtered" % (PARA, SIZE), "refine_blend_reweight_snowy_%s_%s_cn_filtered" % (PARA, SIZE), "replace_blend_reweight_rainy_%s_%s_cn_filtered" % (PARA, SIZE)]

PARQUETS = ["replace_blend_reweight/night/%s/%s_cn_filtered" % (PARAS[0], SIZE), "replace_blend_reweight/rainy/%s/%s_cn_filtered" % (PARAS[0], SIZE), "replace_blend_reweight/foggy/%s/%s_cn_filtered" % (PARAS[0], SIZE), "refine_blend_reweight/snowy/%s/%s_cn_filtered" % (PARAS[0], SIZE)]


PARQUET_ROOT = "/mnt/ve_share/songyuhao/generation/data/train/diffusions/parquet_v2"
PARQUET_PATH = "%s/all_0.50_2.00/%d_filtered" % (PARQUET_ROOT, SIZE*len(PARQUETS))
os.makedirs(PARQUET_PATH, exist_ok=True)
PARQUET_PATH = "%s/pcn.parquet" % PARQUET_PATH

files = ["%s/%s/pcn.parquet" % (PARQUET_ROOT, _) for _ in PARQUETS]

with pq.ParquetWriter(PARQUET_PATH, schema=pq.ParquetFile(files[0]).schema_arrow) as writer:
    for file in files:
        writer.write_table(pq.read_table(file))
        
print(PARQUET_PATH)
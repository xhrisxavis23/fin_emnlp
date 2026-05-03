import qlib

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

from qlib.data import D

instruments = D.instruments()
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2008-12-29":].sort_index()

data.to_hdf("./daily_pv_all.h5", key="data")


fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
debug_data = D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day")

base_instruments = data.reset_index()["instrument"].unique()
debug_instruments = set(debug_data.index.get_level_values("instrument").unique())
selected_instruments = [instrument for instrument in base_instruments if instrument in debug_instruments][:100]

debug_data = debug_data[debug_data.index.get_level_values("instrument").isin(selected_instruments)]
debug_data = debug_data.reorder_levels(["datetime", "instrument"]).sort_index()

debug_data.to_hdf("./daily_pv_debug.h5", key="data")

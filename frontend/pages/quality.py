import streamlit as st

from backend.services.data_quality_service import channel_catalog, descriptive_statistics, missing_statistics, sensor_outages
from frontend.components import page_header, section


def render(full_data, filtered_data, report) -> None:
    page_header("Transparency and provenance", "Data Quality & Dataset Explorer", "Inspect coverage, missing observations, outages and standardized measurement channels.")
    cols=st.columns(4)
    cols[0].metric("Dataset rows",f"{report.row_count:,}")
    cols[1].metric("Sampling interval",str(report.median_interval))
    cols[2].metric("Missing timestamps",str(report.missing_intervals))
    cols[3].metric("Duplicate timestamps",str(report.duplicate_timestamps))
    st.caption(f"Coverage: {report.start} to {report.end} • client C001 • region TUN • timezone not supplied")
    section("Measurement Channels")
    st.dataframe(channel_catalog(filtered_data),hide_index=True,width="stretch")
    section("Missing-Value Statistics","Original missingness is retained even where bounded interpolation produced an analysis value.")
    st.dataframe(missing_statistics(filtered_data),hide_index=True,width="stretch")
    section("Sensor Outages")
    minimum=st.number_input("Minimum outage duration to display (minutes)",min_value=1,max_value=1440,value=30,step=1)
    outages=sensor_outages(filtered_data,int(minimum))
    st.dataframe(outages,hide_index=True,width="stretch")
    section("Descriptive Statistics")
    st.dataframe(descriptive_statistics(filtered_data),hide_index=True,width="stretch")
    section("Filtered Data Explorer")
    selectable=[c for c in filtered_data.columns if not c.endswith("_was_missing")]
    defaults=[c for c in ["timestamp","household_power_kw","zone_demand_mw","system_production_mw","zone_pv_production_mw"] if c in selectable]
    selected=st.multiselect("Columns",selectable,default=defaults)
    preview=filtered_data[selected] if selected else filtered_data.iloc[:,0:0]
    st.dataframe(preview.head(5000),hide_index=True,width="stretch")
    st.download_button("Export filtered standardized data",preview.to_csv(index=False).encode("utf-8"),"filtered_energy_data.csv","text/csv")
    st.caption("Raw CSV data is preserved on disk. Standardized analysis data interpolates only internal continuous-sensor gaps up to 180 minutes; longer outages remain missing and are not filled with zero.")

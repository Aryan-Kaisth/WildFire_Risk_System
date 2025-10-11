from pydantic import BaseModel, Field, computed_field
from typing import Annotated
from datetime import date

class TextRequest(BaseModel):
    # --- Original Fields ---
    latitude: Annotated[float, Field(..., description="Latitude of the location in decimal degrees")]
    longitude: Annotated[float, Field(..., description="Longitude of the location in decimal degrees")]
    datetime: Annotated[date, Field(..., description="Date and time of the recorded data")]
    pr: Annotated[float, Field(..., description="Precipitation amount (mm/day)")]
    rmax: Annotated[float, Field(..., description="Maximum relative humidity (%)")]
    rmin: Annotated[float, Field(..., description="Minimum relative humidity (%)")]
    sph: Annotated[float, Field(..., description="Specific humidity (kg/kg)")]
    srad: Annotated[float, Field(..., description="Solar radiation (kJ/m²)")]
    tmmn: Annotated[float, Field(..., description="Minimum temperature (°C)")]
    tmmx: Annotated[float, Field(..., description="Maximum temperature (°C)")]
    vs: Annotated[float, Field(..., description="Wind speed at 10 meters (m/s)")]
    bi: Annotated[float, Field(..., description="Burning index, an indicator of fire intensity potential")]
    fm100: Annotated[float, Field(..., description="100-hour fuel moisture (%)")]
    fm1000: Annotated[float, Field(..., description="1000-hour fuel moisture (%)")]
    erc: Annotated[float, Field(..., description="Energy Release Component, measures potential fire energy (unitless)")]
    etr: Annotated[float, Field(..., description="Reference evapotranspiration (mm/day)")]
    pet: Annotated[float, Field(..., description="Potential evapotranspiration (mm/day)")]
    vpd: Annotated[float, Field(..., description="Vapor pressure deficit (kPa)")]

    # --- Computed datetime fields ---
    @computed_field(description="Year extracted from datetime")
    @property
    def year(self) -> int:
        return self.datetime.year

    @computed_field(description="Month extracted from datetime")
    @property
    def month(self) -> int:
        return self.datetime.month

    @computed_field(description="Day extracted from datetime")
    @property
    def day(self) -> int:
        return self.datetime.day

    @computed_field(description="Day of week (0=Monday, 6=Sunday)")
    @property
    def dayofweek(self) -> int:
        return self.datetime.weekday()

    @computed_field(description="Quarter of the year (1-4)")
    @property
    def quarter(self) -> int:
        return (self.datetime.month - 1) // 3 + 1

    @computed_field(description="Day of year (1–366)")
    @property
    def dayofyear(self) -> int:
        return self.datetime.timetuple().tm_yday

    @computed_field(description="ISO week number of the year")
    @property
    def weekofyear(self) -> int:
        return self.datetime.isocalendar().week

    @computed_field(description="1 if weekend (Saturday or Sunday), else 0")
    @property
    def is_weekend(self) -> int:
        return int(self.datetime.weekday() >= 5)

    # --- Computed meteorological fields ---
    @computed_field(description="Temperature range (tmmx - tmmn)")
    @property
    def trange(self) -> float:
        return self.tmmx - self.tmmn

    @computed_field(description="Rainfall range (rmax - rmin)")
    @property
    def rrange(self) -> float:
        return self.rmax - self.rmin

    @computed_field(description="Fuel moisture ratio (fm100 / fm1000)")
    @property
    def fm_ratio(self) -> float:
        return self.fm100 / self.fm1000 if self.fm1000 != 0 else 0.0

    @computed_field(description="Absolute water deficit (pet - etr, mm/day)")
    @property
    def pet_minus_etr(self) -> float:
        return self.pet - self.etr

    @computed_field(description="Temperature × radiation stress indicator (trange * srad)")
    @property
    def trange_srad(self) -> float:
        return (self.tmmx - self.tmmn) * self.srad

    @computed_field(description="Fire risk indicator (vpd × tmmx)")
    @property
    def vpd_tmmx(self) -> float:
        return self.vpd * self.tmmx

    @computed_field(description="Ignition potential (fm100 × wind speed)")
    @property
    def fm_wind(self) -> float:
        return self.fm100 * self.vs

    @computed_field(description="Rainfall relative to max rainfall (pr / rmax)")
    @property
    def pr_rmax_ratio(self) -> float:
        return self.pr / self.rmax if self.rmax != 0 else 0.0

    @computed_field(description="Difference between fine and coarse fuel moisture (fm100 - fm1000)")
    @property
    def fm_diff(self) -> float:
        return self.fm100 - self.fm1000

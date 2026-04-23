from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    datapath: Path = Path("./data")
    target_continent: str | None = "Asia"
    niche_type: str = "gdp_per_capita_2018"
    niche_method: str = "zscore_log"
    verbose: bool = True
    num_particles: int = 1000
    sigma: float = 1.0

    @property
    def pop_path(self) -> Path:
        return self.datapath / "population/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv"

    @property
    def flow_path(self) -> Path:
        return self.datapath / "migrations/international_migration_flow.csv"

    @property
    def gdp_path(self) -> Path:
        return self.datapath / "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_46.csv"

    @property
    def migration_path(self) -> Path:
        return self.datapath / "migration_stock_2018.csv"

    @property
    def hdi_path(self) -> Path:
        return self.datapath / "hdi_2020_clean.csv"

    @property
    def niche_path(self) -> Path:
        if self.niche_type == "gdp_per_capita_2018":
            return self.gdp_path
        raise ValueError(f"Unknown niche_type: {self.niche_type}")

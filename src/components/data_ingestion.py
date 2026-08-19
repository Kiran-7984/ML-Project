from pathlib import Path
import pandas as pd
import logging


class DataIngestionConfig:
    """
    Configuration for reading the raw LTPP Excel workbook.
    """

    def __init__(self):
        self.raw_data_path = Path("data/raw/Bucket_114229.xlsx")


class DataIngestion:
    """
    Reads only the LTPP sheets required for the IRI deterioration project.
    """

    REQUIRED_SHEETS = {
        "iri": "MON_HSS_PROFILE_SECTION",
        "precipitation": "CLM_VWS_PRECIP_ANNUAL",
        "temperature": "CLM_VWS_TEMP_ANNUAL",
        "traffic": "TRF_TREND_1",
        "history": "PROJECT_HIST_AGE_EXP",
        "section": "SECTION_GENERAL_EXP",
        "layers": "TST_L05B",
    }

    def __init__(self):
        self.config = DataIngestionConfig()
        logging.basicConfig(level=logging.INFO)

    def _check_file(self):
        if not self.config.raw_data_path.exists():
            raise FileNotFoundError(
                f"Raw dataset not found: {self.config.raw_data_path}"
            )

    def _read_sheet(self, sheet_name, columns):
        logging.info(f"Reading sheet: {sheet_name}")

        df = pd.read_excel(
            self.config.raw_data_path,
            sheet_name=sheet_name,
            usecols=columns,
            engine="openpyxl",
        )

        logging.info(
            f"{sheet_name}: {len(df)} rows, {len(df.columns)} columns"
        )

        return df

    def initiate_data_ingestion(self):

        self._check_file()

        data = {}

        data["iri"] = self._read_sheet(
            self.REQUIRED_SHEETS["iri"],
            [
                "VISIT_DATE",
                "CONSTRUCTION_NO",
                "STATE_CODE",
                "STATE_CODE_EXP",
                "VISIT_NO",
                "SHRP_ID",
                "MRI",
            ],
        )

        data["precipitation"] = self._read_sheet(
            self.REQUIRED_SHEETS["precipitation"],
            [
                "STATE_CODE",
                "SHRP_ID",
                "YEAR",
                "TOTAL_ANN_PRECIP",
            ],
        )

        data["temperature"] = self._read_sheet(
            self.REQUIRED_SHEETS["temperature"],
            [
                "STATE_CODE",
                "SHRP_ID",
                "YEAR",
                "MEAN_ANN_TEMP_AVG",
                "FREEZE_INDEX_YR",
                "FREEZE_THAW_YR",
            ],
        )

        data["traffic"] = self._read_sheet(
            self.REQUIRED_SHEETS["traffic"],
            [
                "STATE_CODE",
                "SHRP_ID",
                "CONSTRUCTION_NO",
                "YEAR",
                "AADTT_ALL_TRUCKS_TREND",
            ],
        )

        data["history"] = self._read_sheet(
            self.REQUIRED_SHEETS["history"],
            [
                "STATE_CODE",
                "SHRP_ID",
                "CONSTRUCTION_DATE",
                "TRAFFIC_OPEN_DATE",
                "ORIGINAL_NO_LANES",
                "FINAL_NO_LANES",
                "LANE_ADDED_NO",
            ],
        )

        data["section"] = self._read_sheet(
            self.REQUIRED_SHEETS["section"],
            [
                "STATE_CODE",
                "SHRP_ID",
                "MONITORED_LANE",
                "LANE_WIDTH",
                "SECTION_LENGTH",
                "SPEED_LIMIT",
                "DIRECTION_OF_TRAVEL_EXP",
            ],
        )

        data["layers"] = self._read_sheet(
            self.REQUIRED_SHEETS["layers"],
            [
                "STATE_CODE",
                "SHRP_ID",
                "CONSTRUCTION_NO",
                "LAYER_TYPE_EXP",
                "REPR_THICKNESS",
            ],
        )

        logging.info("Data ingestion completed successfully.")

        # with pd.ExcelWriter(
        #     "data/raw/final_data.xlsx",
        #     engine="openpyxl"
        # ) as writer:

        #     for name, df in data.items():
        #         df.to_excel(
        #             writer,
        #             sheet_name=name,
        #             index=False
        #         )
        for name, df in data.items():
            df.to_csv(
                f"data/raw/{name}.csv",
                index=False
            )

        return data


if __name__ == "__main__":

    ingestion = DataIngestion()

    datasets = ingestion.initiate_data_ingestion()

    for name, df in datasets.items():
        print(f"{name}: {df.shape}")
import pandas as pd
from pathlib import Path


class DataTransformation:

    def __init__(self):
        self.raw = Path("data/raw")
        self.output = Path(
            "data/processed/iri_deterioration.csv"
        )

    def transform(self):

        iri = pd.read_csv(self.raw / "iri.csv")
        rain = pd.read_csv(self.raw / "precipitation.csv")
        temp = pd.read_csv(self.raw / "temperature.csv")
        traffic = pd.read_csv(self.raw / "traffic.csv")
        history = pd.read_csv(self.raw / "history.csv")
        section = pd.read_csv(self.raw / "section.csv")
        layers = pd.read_csv(self.raw / "layers.csv")

        keys = ["STATE_CODE", "SHRP_ID"]
        ckeys = keys + ["CONSTRUCTION_NO"]

        iri["VISIT_DATE"] = pd.to_datetime(iri["VISIT_DATE"])
        iri["MRI"] = pd.to_numeric(iri["MRI"], errors="coerce")

        iri = iri.dropna(
            subset=ckeys + ["VISIT_DATE", "MRI"]
        )

        iri = iri.sort_values(
            ckeys + ["VISIT_DATE"]
        )

        iri["VISIT_YEAR"] = iri["VISIT_DATE"].dt.year

        iri["NEXT_MRI"] = (
            iri.groupby(ckeys)["MRI"].shift(-1)
        )

        iri["NEXT_VISIT_DATE"] = (
            iri.groupby(ckeys)["VISIT_DATE"].shift(-1)
        )

        iri["YEARS_TO_NEXT"] = (
            iri["NEXT_VISIT_DATE"] - iri["VISIT_DATE"]
        ).dt.days / 365.25

        climate = rain.merge(
            temp,
            on=keys + ["YEAR"],
            how="outer"
        )

        iri = iri.merge(
            climate,
            left_on=keys + ["VISIT_YEAR"],
            right_on=keys + ["YEAR"],
            how="left"
        ).drop(columns="YEAR")

        iri = iri.merge(
            traffic,
            left_on=ckeys + ["VISIT_YEAR"],
            right_on=ckeys + ["YEAR"],
            how="left"
        ).drop(columns="YEAR")

        history["CONSTRUCTION_DATE"] = pd.to_datetime(
            history["CONSTRUCTION_DATE"],
            errors="coerce"
        )

        iri = iri.merge(
            history,
            on=keys,
            how="left"
        )

        iri["PAVEMENT_AGE"] = (
            iri["VISIT_DATE"] - iri["CONSTRUCTION_DATE"]
        ).dt.days / 365.25

        iri = iri.merge(
            section,
            on=keys,
            how="left"
        )

        layers["REPR_THICKNESS"] = pd.to_numeric(
            layers["REPR_THICKNESS"],
            errors="coerce"
        ).fillna(0)

        layers["AC"] = layers["BASE"] = layers["SUBBASE"] = 0

        iri_types = layers["LAYER_TYPE_EXP"]

        layers.loc[
            iri_types == "Asphalt concrete layer", "AC"
        ] = layers["REPR_THICKNESS"]

        layers.loc[
            iri_types.str.contains("base", case=False, na=False)
            & ~iri_types.str.contains("subbase", case=False, na=False),
            "BASE"
        ] = layers["REPR_THICKNESS"]

        layers.loc[
            iri_types.str.contains("subbase", case=False, na=False),
            "SUBBASE"
        ] = layers["REPR_THICKNESS"]

        layers = layers.groupby(ckeys)[
            ["AC", "BASE", "SUBBASE"]
        ].sum().reset_index()

        layers = layers.rename(columns={
            "AC": "AC_THICKNESS",
            "BASE": "BASE_THICKNESS",
            "SUBBASE": "SUBBASE_THICKNESS"
        })

        iri = iri.merge(
            layers,
            on=ckeys,
            how="left"
        )

        iri["MRI_CHANGE"] = (
            iri["NEXT_MRI"] - iri["MRI"]
        )

        iri["IRI_DETERIORATION_RATE"] = (
            iri["MRI_CHANGE"] / iri["YEARS_TO_NEXT"]
        )

        iri = iri.dropna(
            subset=["NEXT_MRI", "YEARS_TO_NEXT"]
        )

        iri = iri[iri["YEARS_TO_NEXT"] > 0]

        self.output.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        columns = [
        "STATE_CODE",
        "SHRP_ID",
        "CONSTRUCTION_NO",
        "MRI",
        "NEXT_MRI",
        "NEXT_VISIT_DATE",
        "YEARS_TO_NEXT",
        "AADTT_ALL_TRUCKS_TREND",
        "TOTAL_ANN_PRECIP",
        "MEAN_ANN_TEMP_AVG",
        "FREEZE_INDEX_YR",
        "FREEZE_THAW_YR",
        "PAVEMENT_AGE",
        "AC_THICKNESS",
        "BASE_THICKNESS",
        "SUBBASE_THICKNESS",
        "LANE_WIDTH",
        "SECTION_LENGTH",
        "DIRECTION_OF_TRAVEL_EXP"
        ]

        columns = [
            col for col in columns
            if col in iri.columns
        ]

        iri = iri[columns]

        iri.to_csv(
            self.output,
            index=False
        )

        print("Transformation completed!")
        print("Final shape:", iri.shape)

        return iri
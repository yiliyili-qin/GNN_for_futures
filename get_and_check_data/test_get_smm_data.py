import hashlib
import unittest
from unittest.mock import Mock

from get_smm_data import (
    SmmClient,
    SmmError,
    download_quota_catalog,
    download_quotas,
    normalize_data,
)


class SmmClientTests(unittest.TestCase):
    def test_auth_hashes_password_and_uses_documented_fields(self) -> None:
        response = Mock()
        response.json.return_value = {"code": 0, "data": {"token": "token-value"}}
        response.raise_for_status.return_value = None
        session = Mock()
        session.request.return_value = response

        client = SmmClient(
            user_name="account",
            password="plain-password",
            session=session,
        )

        self.assertEqual(client.auth(), "token-value")
        request = session.request.call_args
        self.assertEqual(request.args[:2], ("POST", "https://datapro-api.smm.cn/dapi/user/auth"))
        self.assertEqual(request.kwargs["data"]["user_name"], "account")
        self.assertEqual(request.kwargs["data"]["source"], "datapro")
        self.assertEqual(
            request.kwargs["data"]["password"],
            hashlib.md5(b"plain-password").hexdigest(),
        )
        self.assertNotIn("plain-password", str(request.kwargs["data"]))

    def test_download_batches_more_than_one_hundred_ids(self) -> None:
        client = Mock()
        client.get_data_origin.side_effect = [
            {"code": 0, "data": []},
            {"code": 0, "data": []},
        ]
        quota_ids = [f"a{index}" for index in range(101)]

        result = download_quotas(client, quota_ids, "2024-01-01", "2024-01-31")

        self.assertTrue(result.normalized.empty)
        self.assertEqual(client.get_data_origin.call_count, 2)
        self.assertEqual(len(client.get_data_origin.call_args_list[0].args[0]), 100)
        self.assertEqual(len(client.get_data_origin.call_args_list[1].args[0]), 1)

    def test_download_rejects_unauthorized_quota(self) -> None:
        client = Mock()
        client.get_data_origin.return_value = {
            "code": 0,
            "data": [
                {
                    "quota_id": "a10002884",
                    "perm_state": -1,
                    "state_desc": "no authorize",
                    "data": None,
                }
            ],
        }

        with self.assertRaisesRegex(SmmError, "a10002884"):
            download_quotas(client, ["a10002884"], "2024-01-01", "2024-01-31")


class NormalizeDataTests(unittest.TestCase):
    def test_normalizes_series_metadata_and_numeric_values(self) -> None:
        payloads = [
            {
                "code": 0,
                "data": [
                    {
                        "quota_id": "a10002884",
                        "quota_name": "sample indicator",
                        "unit": "yuan/ton",
                        "frequency": "daily",
                        "data": [
                            {"date": "2024-01-02", "value": "12.5"},
                            {"date": "2024-01-01", "value": "11.0"},
                        ],
                    }
                ],
            }
        ]

        frame = normalize_data(payloads)

        self.assertEqual(frame["quota_id"].tolist(), ["a10002884", "a10002884"])
        self.assertEqual(frame["value"].tolist(), [11.0, 12.5])
        self.assertEqual(frame["frequency"].tolist(), ["daily", "daily"])
        self.assertEqual(frame["date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-01", "2024-01-02"])

    def test_catalog_filters_name_and_flattens_update_rule(self) -> None:
        client = Mock()
        client.list_quotas.return_value = {
            "page_count": 1,
            "data": [
                {
                    "quota_id": "s20015437",
                    "quota_name": "SMM 1# electrolytic copper average price",
                    "unit": "yuan/ton",
                    "frequency": "daily",
                    "type_all": "basic metals;copper;spot price",
                    "update_rule": {
                        "release_time": "12:00:00",
                        "rule_desc": "updated daily",
                    },
                },
                {
                    "quota_id": "other",
                    "quota_name": "aluminum inventory",
                },
            ],
        }

        catalog = download_quota_catalog(client, keywords=["copper"])

        self.assertEqual(catalog["quota_id"].tolist(), ["s20015437"])
        self.assertEqual(catalog["release_time"].tolist(), ["12:00:00"])


if __name__ == "__main__":
    unittest.main()
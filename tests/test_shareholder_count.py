"""Offline parser, failure and cross-instance cache checks for holder counts."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import Mock, patch

import requests

from data_provider.shareholder_count import ShareholderCountProvider


AS_OF = date(2026, 10, 18)
NOW = datetime(2026, 10, 18, 1, 0, tzinfo=timezone.utc)
MODULE = "data_provider.shareholder_count"

# Same business envelope/fields as the three real API probes saved outside Git.
REAL_RESPONSE = {
    "version": "84231e3011f6c302350ea1ed9e23b904",
    "result": {
        "pages": 323,
        "data": [{
            "SECURITY_CODE": "002727", "SECURITY_NAME_ABBR": "一心堂",
            "HOLDER_NUM": 39518, "END_DATE": "2026-06-30 00:00:00",
            "HOLD_NOTICE_DATE": "2026-08-27 00:00:00",
        }],
        "count": 323,
    },
    "success": True, "message": "ok", "code": 0,
}


def api_response(body=None):
    response = Mock()
    response.json.return_value = deepcopy(REAL_RESPONSE if body is None else body)
    return response


class ShareholderCountProviderTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name)
        self.provider = ShareholderCountProvider(self.cache)
        self.clock = patch(f"{MODULE}._utc_now", return_value=NOW).start()
        self.addCleanup(patch.stopall)

    def fetch(self, body=None, code="002727", as_of=AS_OF):
        with patch(f"{MODULE}.requests.get", return_value=api_response(body)) as get:
            result = self.provider.get_latest(code, as_of)
        return result, get

    def write_cache(self, payload=None, as_of=AS_OF):
        if payload is None:
            payload = self.valid_payload()
        path = self.cache / f"002727_{AS_OF.isoformat()}.json"
        path.write_text(json.dumps({"as_of": as_of.isoformat(), "data": payload}), encoding="utf-8")
        return path

    def valid_payload(self):
        return {
            "code": "002727", "count": 39518, "end_date": "2026-06-30",
            "announcement_date": "2026-08-27", "source": "东方财富",
            "source_url": "https://data.eastmoney.com/gdhs/detail/002727.html",
            "fetched_at": NOW.isoformat(),
        }

    def test_parses_verified_real_response_without_changing_unit(self):
        result, get = self.fetch()
        self.assertEqual(result, self.valid_payload())
        get.assert_called_once()
        self.assertEqual(get.call_args.args, ("https://datacenter-web.eastmoney.com/api/data/v1/get",))
        self.assertEqual(get.call_args.kwargs["timeout"], (3, 5))
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["reportName"], "RPT_HOLDERNUM_DET")
        self.assertEqual(params["sortColumns"], "END_DATE,HOLD_NOTICE_DATE")
        self.assertEqual(params["sortTypes"], "-1,-1")
        self.assertEqual(params["pageSize"], "1")
        self.assertEqual(params["filter"],
                         '(SECURITY_CODE="002727")(END_DATE<=\'2026-10-18\')'
                         '(HOLD_NOTICE_DATE<=\'2026-10-18\')')

    def test_all_three_live_probe_payloads_parse(self):
        for code, name, count in (("002727", "一心堂", 39518),
                                  ("603173", "福斯达", 11850), ("603992", "松霖科技", 14014)):
            with self.subTest(code=code):
                body = deepcopy(REAL_RESPONSE)
                body["result"]["data"][0].update(SECURITY_CODE=code, SECURITY_NAME_ABBR=name, HOLDER_NUM=count)
                result, _ = self.fetch(body, code)
                self.assertEqual(result["code"], code)
                self.assertEqual(result["count"], count)
                self.assertEqual(result["end_date"], "2026-06-30")
                self.assertEqual(result["announcement_date"], "2026-08-27")

    def test_accepts_correct_explicit_exchange_and_keeps_six_digit_code(self):
        for code in ("SZ002727", "sz.002727", "002727.SZ", " 002727 "):
            with self.subTest(code=code):
                result, _ = self.fetch(code=code)
                self.assertEqual(result["code"], "002727")

    def test_conflicting_exchange_index_etf_and_non_a_codes_make_no_request(self):
        for code in ("SH002727", "002727.SH", "SH000001", "SZ603173", "BJ603992",
                     "HK00700", "AAPL", "510300", "999999", "2727", "../../002727", 2727, None):
            with self.subTest(code=code):
                result, get = self.fetch(code=code)
                self.assertIsNone(result)
                get.assert_not_called()

    def test_future_or_non_date_as_of_makes_no_request(self):
        for as_of in (date(2026, 10, 19), "2026-10-18", NOW, None):
            with self.subTest(as_of=as_of):
                result, get = self.fetch(as_of=as_of)
                self.assertIsNone(result)
                get.assert_not_called()

    def test_uses_beijing_date_when_utc_is_still_previous_day(self):
        self.clock.return_value = datetime(2026, 10, 17, 17, 0, tzinfo=timezone.utc)
        result, get = self.fetch()
        self.assertIsNotNone(result)
        get.assert_called_once()

    def test_invalid_counts_never_become_data_or_a_success_cache(self):
        for value in (True, False, 0, -1, 1.2, 39518.0, float("nan"), float("inf"), "39518", None):
            with self.subTest(count=value):
                body = deepcopy(REAL_RESPONSE)
                body["result"]["data"][0]["HOLDER_NUM"] = value
                result, _ = self.fetch(body)
                self.assertIsNone(result)
                self.assertEqual(list(self.cache.iterdir()), [])

    def test_wrong_stock_missing_or_invalid_dates_are_rejected(self):
        for field, value in (("SECURITY_CODE", "603173"), ("SECURITY_CODE", 2727),
                             ("END_DATE", "2026-02-30"), ("END_DATE", None),
                             ("END_DATE", "2026-10-19"), ("END_DATE", "2026-09-01"),
                             ("HOLD_NOTICE_DATE", "not-a-date"), ("HOLD_NOTICE_DATE", None),
                             ("HOLD_NOTICE_DATE", "2026-10-19"), ("HOLD_NOTICE_DATE", "2026-06-29")):
            with self.subTest(field=field, value=value):
                body = deepcopy(REAL_RESPONSE)
                body["result"]["data"][0][field] = value
                result, _ = self.fetch(body)
                self.assertIsNone(result)
                self.assertEqual(list(self.cache.iterdir()), [])

    def test_historical_cutoff_does_not_accept_a_later_disclosure(self):
        # June statistics existed by August 26, but their August 27 publication
        # was not yet available. The client rejects even if a server ignores filters.
        result, get = self.fetch(as_of=date(2026, 8, 26))
        self.assertIsNone(result)
        self.assertIn("HOLD_NOTICE_DATE<='2026-08-26'", get.call_args.kwargs["params"]["filter"])

    def test_verified_historical_cutoff_response_uses_prior_published_count(self):
        body = deepcopy(REAL_RESPONSE)
        body["result"]["data"][0].update(
            HOLDER_NUM=41801, END_DATE="2026-03-31 00:00:00", HOLD_NOTICE_DATE="2026-04-29 00:00:00",
        )
        result, _ = self.fetch(body, as_of=date(2026, 8, 26))
        self.assertEqual(result["count"], 41801)
        self.assertEqual(result["end_date"], "2026-03-31")

    def test_no_disclosures_and_bad_business_envelopes_return_none(self):
        bodies = [[], "bad", {"success": False, "code": 0}, {"success": True, "code": 1},
                  {"success": True, "code": False}, {"success": True, "code": 0, "result": None}]
        for data in ([], None, {}, [None]):
            body = deepcopy(REAL_RESPONSE)
            body["result"]["data"] = data
            bodies.append(body)
        for body in bodies:
            with self.subTest(body=body):
                result, get = self.fetch(body)
                self.assertIsNone(result)
                get.assert_called_once()
                self.assertEqual(list(self.cache.iterdir()), [])

    def test_http_connection_timeout_and_invalid_json_are_fail_open_without_retry(self):
        for error in (requests.Timeout("slow"), requests.ConnectionError("offline")):
            with self.subTest(error=error):
                with patch(f"{MODULE}.requests.get", side_effect=error) as get:
                    with self.assertLogs(MODULE, level="WARNING"):
                        self.assertIsNone(self.provider.get_latest("002727", AS_OF))
                get.assert_called_once()
        for field, error in (("raise_for_status", requests.HTTPError("503")),
                             ("json", ValueError("not json"))):
            with self.subTest(field=field):
                response = api_response()
                getattr(response, field).side_effect = error
                with patch(f"{MODULE}.requests.get", return_value=response) as get:
                    self.assertIsNone(self.provider.get_latest("002727", AS_OF))
                get.assert_called_once()
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_failures_are_not_cached_and_next_call_can_succeed(self):
        with patch(f"{MODULE}.requests.get", side_effect=requests.Timeout):
            self.assertIsNone(self.provider.get_latest("002727", AS_OF))
        self.assertEqual(list(self.cache.iterdir()), [])
        result, get = self.fetch()
        self.assertEqual(result["count"], 39518)
        get.assert_called_once()

    def test_fresh_disk_cache_is_shared_by_a_new_provider_while_offline(self):
        result, _ = self.fetch()
        self.clock.return_value = NOW + timedelta(minutes=59)
        separate = ShareholderCountProvider(self.cache)
        with patch(f"{MODULE}.requests.get", side_effect=requests.ConnectionError("offline")) as get:
            self.assertEqual(separate.get_latest("002727", AS_OF), result)
        get.assert_not_called()
        self.assertEqual(len(list(self.cache.glob("*.json"))), 1)
        self.assertEqual(list(self.cache.glob("*.tmp")), [])

    def test_expired_cache_does_not_get_returned_when_offline(self):
        self.write_cache()
        self.clock.return_value = NOW + timedelta(hours=1)
        with patch(f"{MODULE}.requests.get", side_effect=requests.Timeout) as get:
            self.assertIsNone(self.provider.get_latest("002727", AS_OF))
        get.assert_called_once()

    def test_invalid_cache_fields_are_revalidated_before_use(self):
        invalid = [
            ("code", "603173"), ("count", True), ("count", 1.2), ("count", 0),
            ("end_date", "not-a-date"), ("end_date", "2026-10-19"),
            ("announcement_date", "2026-06-01"), ("announcement_date", "2026-10-19"),
            ("source", "other"), ("source_url", "https://example.org/track"),
            ("fetched_at", (NOW + timedelta(seconds=1)).isoformat()),
            ("fetched_at", (NOW - timedelta(hours=1)).isoformat()),
            ("fetched_at", NOW.replace(tzinfo=None).isoformat()), ("fetched_at", "bad"),
        ]
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                payload = self.valid_payload()
                payload[field] = value
                self.write_cache(payload)
                with patch(f"{MODULE}.requests.get", side_effect=requests.Timeout) as get:
                    self.assertIsNone(self.provider.get_latest("002727", AS_OF))
                get.assert_called_once()

    def test_corrupt_or_wrong_as_of_cache_is_not_used(self):
        for text in ("not json", "[]", "null", json.dumps({"as_of": "2026-10-17", "data": self.valid_payload()})):
            with self.subTest(cache=text):
                path = self.cache / "002727_2026-10-18.json"
                path.write_text(text, encoding="utf-8")
                result, get = self.fetch()
                self.assertEqual(result["count"], 39518)
                get.assert_called_once()

    def test_cache_cannot_claim_disclosure_published_after_its_fetch_day(self):
        self.clock.return_value = datetime(2026, 10, 17, 16, 10, tzinfo=timezone.utc)
        payload = self.valid_payload()
        payload["announcement_date"] = "2026-10-18"
        payload["fetched_at"] = datetime(2026, 10, 17, 15, 30, tzinfo=timezone.utc).isoformat()
        self.write_cache(payload)
        with patch(f"{MODULE}.requests.get", side_effect=requests.Timeout) as get:
            self.assertIsNone(self.provider.get_latest("002727", AS_OF))
        get.assert_called_once()

    def test_as_of_is_part_of_the_cache_key(self):
        self.fetch()
        result, get = self.fetch(as_of=date(2026, 10, 17))
        self.assertIsNotNone(result)
        get.assert_called_once()
        self.assertEqual(len(list(self.cache.glob("*.json"))), 2)

    def test_unreadable_cache_does_not_prevent_fetching_data(self):
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            result, get = self.fetch()
        self.assertEqual(result["count"], 39518)
        get.assert_called_once()

    def test_cache_directory_and_replace_failures_do_not_hide_acquired_data(self):
        for method in ("mkdir", "replace"):
            with self.subTest(method=method):
                with patch.object(Path, method, side_effect=PermissionError("denied")):
                    result, _ = self.fetch()
                self.assertEqual(result["count"], 39518)
                self.assertEqual(list(self.cache.glob("*.tmp")), [])

    def test_concurrent_writers_keep_a_complete_reusable_json_cache(self):
        barrier = Barrier(4)

        def same_time_response(*args, **kwargs):
            # No writer can populate the cache before every worker has observed
            # its absence. This exercises concurrent writes to the same key.
            barrier.wait(timeout=3)
            return api_response()

        with patch(f"{MODULE}.requests.get", side_effect=same_time_response) as get:
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: self.provider.get_latest("002727", AS_OF), range(4)))
        self.assertEqual(get.call_count, 4)
        self.assertEqual(results, [self.valid_payload()] * 4)
        cached = json.loads((self.cache / "002727_2026-10-18.json").read_text(encoding="utf-8"))
        self.assertEqual(cached["data"], self.valid_payload())
        self.assertEqual(list(self.cache.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()

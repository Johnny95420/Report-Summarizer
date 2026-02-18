import atexit
import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from concurrent.futures import (
    TimeoutError as ExecutorTimeout,
)
from queue import Queue

import trafilatura
import undetected_chromedriver as uc
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query


# =========================
# Safe execute_script with timeout
# =========================
def safe_execute_script(driver, script, timeout=5):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(driver.execute_script, script)
        try:
            return future.result(timeout=timeout)
        except ExecutorTimeout:
            print(f"[DriverPool] execute_script timeout after {timeout}s.")
            raise Exception("execute_script timeout")
        except Exception as e:
            print(f"[DriverPool] execute_script error: {e}")
            raise e


# =========================
# Driver Pool Class
# =========================
class DriverPool:
    def __init__(self, max_drivers=3, retry_create=3):
        self.max_drivers = max_drivers
        self.retry_create = retry_create
        self.pool = Queue(maxsize=max_drivers)
        self.lock = threading.Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        for _ in range(self.max_drivers):
            driver = self._create_driver_with_retry()
            self.pool.put(driver)
        print(f"[DriverPool] Initialized with {self.max_drivers} drivers.")

    def _create_driver_with_retry(self):
        for attempt in range(1, self.retry_create + 1):
            try:
                return self._create_driver()
            except Exception as e:
                print(f"[DriverPool] Attempt {attempt} to create driver failed: {e}")
                time.sleep(1)
        raise Exception("[DriverPool] Failed to create driver after retries.")

    def _create_driver(self):
        options = uc.ChromeOptions()
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--headless")
        driver = uc.Chrome(options=options)
        print("[DriverPool] New undetected_chromedriver created.")
        return driver

    def _health_check(self, driver, timeout=5):
        try:
            result = safe_execute_script(driver, "return 1+1;", timeout=timeout)
            return result == 2
        except Exception as e:
            print(f"[DriverPool] Health check failed: {e}")
            return False

    def acquire(self):
        driver = self.pool.get()
        print(f"[DriverPool] Acquired driver. {self.pool.qsize()} remaining.")

        if not self._health_check(driver, timeout=5):
            print("[DriverPool] Driver unhealthy during acquire. Restarting...")
            self.restart_driver(driver)
            driver = self.pool.get()

        return driver

    def release(self, driver):
        if not self._health_check(driver, timeout=5):
            print("[DriverPool] Driver unhealthy during release. Restarting...")
            self.restart_driver(driver)
        else:
            self.pool.put(driver)
            print(f"[DriverPool] Released driver. {self.pool.qsize()} available.")

    def restart_driver(self, driver):
        print("[DriverPool] Restarting driver due to fatal error...")
        self._restart_driver(driver)

        try:
            new_driver = self._create_driver_with_retry()
            self.pool.put(new_driver)
            print("[DriverPool] Driver restarted and returned to pool.")
        except Exception as e:
            print(f"[DriverPool] Failed to restart driver: {e}")

    def _restart_driver(self, driver):
        try:
            driver.quit()
            print("[DriverPool] Driver quit.")
        except Exception as e:
            print(f"[DriverPool] Driver quit error: {e}")

    def shutdown(self):
        with self.lock:
            while not self.pool.empty():
                driver = self.pool.get()
                self._restart_driver(driver)
            print("[DriverPool] Shutdown complete.")


# =========================
# Global Driver Pool Setup
# =========================
MAX_DRIVERS = 10
driver_pool = DriverPool(max_drivers=MAX_DRIVERS)
atexit.register(driver_pool.shutdown)


# =========================
# Search Function
# =========================
def search(query, time_filter: str = "year", timeout=5):
    driver = driver_pool.acquire()
    page_source = ""
    try:
        driver.set_page_load_timeout(timeout)
        filters = {"day": 'ex1:"ez1"', "week": 'ex1:"ez2"', "month": 'ex1:"ez3"'}
        base_url = f"https://www.bing.com/search?q={query}"
        if time_filter in filters:
            base_url += f"&filters={filters[time_filter]}"

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(driver.get, base_url)
            try:
                future.result(timeout=timeout)
            except ExecutorTimeout:
                print(f"[Search Error] driver.get() timeout after {timeout}s at {base_url}")
                raise Exception(f"driver.get() timeout after {timeout}s")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: driver.page_source)
            try:
                page_source = future.result(timeout=5)
            except ExecutorTimeout:
                print(f"[Search Error] driver.page_source timeout at {base_url}")
                raise Exception("driver.page_source timeout")

    except Exception as e:
        print(f"[Search Error] {e}")
    finally:
        driver_pool.release(driver)

    soup = BeautifulSoup(page_source, "html.parser")
    items = soup.find_all("li", {"class": "b_algo"})
    results = []
    for item in items:
        title_tag = item.find("h2")
        link_tag = title_tag.find("a") if title_tag else None
        body = (
            item.find_all("div", {"class": "b_caption"})[0].get_text()
            if item.find_all("div", {"class": "b_caption"})
            else ""
        )
        if link_tag:
            title = link_tag.get_text(strip=True)
            url = link_tag.get("href")
            results.append({"title": title, "url": url, "content": body})
    return results


# =========================
# Parse Content Function
# =========================
def parse_content(html):
    return trafilatura.extract(html, include_comments=True, include_tables=True, output_format="markdown")


# =========================
# Crawl Single URL Function
# =========================
def crawl_url(result, result_queue, timeout=10):
    driver = driver_pool.acquire()

    try:
        start_time = time.time()
        driver.set_page_load_timeout(timeout)
        driver.implicitly_wait(10)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(driver.get, result["url"])
            try:
                future.result(timeout=timeout)
            except ExecutorTimeout:
                print(f"[Crawl Error] driver.get() timeout after {timeout}s at {result['url']}")
                raise Exception(f"driver.get() timeout after {timeout}s")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: driver.page_source)
            try:
                page_source = future.result(timeout=5)
            except ExecutorTimeout:
                print(f"[Crawl Error] driver.page_source timeout at {result['url']}")
                raise Exception("driver.page_source timeout")

        result["raw_content"] = parse_content(page_source)
        result["crawl_time"] = time.time() - start_time
        print(f"[Crawl] Fetched {result['url']} in {result['crawl_time']:.2f}s")

    except Exception as e:
        result["raw_content"] = None
        result["error"] = str(e)
        print(f"[Crawl Error] {e} at {result['url']}")

        driver_pool.restart_driver(driver)
        driver = None

    finally:
        if driver:
            driver_pool.release(driver)

        result_queue.put(result)


# =========================
# Crawl Search Results
# =========================
def crawl_search_results(search_results, max_threads=3, timeout=8):
    updated_results = []

    def crawl_worker(result):
        result_queue = Queue()
        crawl_url(result, result_queue, timeout=timeout)
        return result_queue.get()

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_result = {executor.submit(crawl_worker, result): result for result in search_results}

        for future in as_completed(future_to_result):
            result = future_to_result[future]
            try:
                data = future.result(timeout=timeout + 10)
                updated_results.append(data)
            except ExecutorTimeout:
                print(f"[Executor Timeout] {result['url']}")
                result["raw_content"] = None
                result["error"] = "Executor timeout"
                updated_results.append(result)
            except Exception as e:
                print(f"[Executor Error] {e} at {result['url']}")
                result["raw_content"] = None
                result["error"] = str(e)
                updated_results.append(result)

    updated_results.sort(key=lambda x: [r["url"] for r in search_results].index(x["url"]))
    return updated_results


# =========================
# FastAPI Setup
# =========================
app = FastAPI()


@app.get("/search_and_crawl")
def search_and_crawl(
    query: str = Query(...),
    max_results: int = Query(3),
    timeout: int = Query(10),
    include_raw_content: bool = False,
):
    retry = 0
    search_results = []
    while search_results == [] and retry <= 5:
        search_results = search(query, timeout=timeout)[:max_results]
        retry += 1

    if include_raw_content:
        search_results = crawl_search_results(search_results, max_threads=MAX_DRIVERS, timeout=timeout)

    return {"results": search_results}


# =========================
# Main Entrypoint
# =========================
if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        driver_pool.shutdown()

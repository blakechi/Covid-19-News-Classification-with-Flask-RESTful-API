"""
    Reference from:
        https://levelup.gitconnected.com/simple-api-using-flask-bc1b7486af88
        https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
        https://www.digitalocean.com/community/tutorials/processing-incoming-request-data-in-flask
"""

import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import requests
import asyncio
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

from server_com import ServerCom


REST_API_URL = "http://localhost:5000/classify"


async def query_data(url, gql_query):
    transport = AIOHTTPTransport(url=url)

    async with Client(
        transport=transport, fetch_schema_from_transport=True,
    ) as session:

        # Execute single query
        query = gql(gql_query)
        result = await session.execute(query)

        return result


if __name__ == "__main__":
    server_com = ServerCom()

    # Use your own query instead
    gql_query = """
        query getArticles {
            Analysis(order_by: {article_id: asc}, where: {type: {_eq: "5"}, Article: {timestamp: {_gte: "2020-04-01", _lte: "2020-06-30"}}}) {
                id
                summary: content(path: "summary[0]")
            }
        }
    """
    result = asyncio.run(query_data(server_com.url, gql_query))['Analysis']

    news_summary = result[0]  # news_summary == pay_load
    req = requests.post(REST_API_URL, json=news_summary).json()

    if req["success"]:
        print(req["prediction"])
    else:
        print("Request failed")
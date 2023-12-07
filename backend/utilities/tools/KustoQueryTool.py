from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoServiceError
from backend.utilities.helpers.EnvHelper import EnvHelper

class KustoQueryTool():
    def __init__(self) -> None:
        self.name = "KustoQuery"

    def query(self,managerAlias, userAlias):
        # Define the connection string
        env_helper: EnvHelper = EnvHelper()
        cluster = env_helper.KUSTO_UAR_ENDPOINT
        database = env_helper.KUSTO_UAR_DATABASE
        app_id = env_helper.KUSTO_UAR_APP_ID
        client_secret = env_helper.KUSTO_UAR_CLIENT_SECRET
        client_id = env_helper.KUSTO_UAR_CLIENT_ID

        kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(cluster,app_id,client_secret,client_id)

        # Create a Kusto client
        client = KustoClient(kcsb)

        # Define the query
        query = f'UarSnapshot | where FTEActiveManagerAlias =~ "{managerAlias}" | where Alias =~ "{userAlias}" | project  ServiceName,ResourceType,ResourceID | take 10'
        print(query)
        # Execute the query
        try:
            response = client.execute_query(database, query)
            print(response.primary_results[0])
            result_str = str(response.primary_results[0])
        except KustoServiceError as error:
            print(f"Failed to execute query: {error}")
            result_str = str(error)
        
        return result_str
    
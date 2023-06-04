import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { RecursiveCharacterTextSplitter} from 'langchain/text_splitter'
import {OpenAI} from 'langchain/llms/openai'
import {loadQAStuffChain} from 'langchain/chains'
import {Document} from 'langchain/document'
import {timeout} from './config'

export const createPineconeIndex = async (
    client,
    indexName,
    vectorDimension
) => {
    console.log(`Creating index ${indexName}...`)
    const existingIndexes = await client.listIndexes()
    if (!existingIndexes.includes(indexName)) {
        console.log(`Creating ${indexName}...`)
        await client.createIndex({
            createRequest: {
                name: indexName,
                dimension: vectorDimension,
                metric: 'cosine'
            }
        });
        console.log(`creating index...please wait for it to finish initializing`)

        await new Promise((resolve) => setTimeout(resolve, timeout));
    } else {
        console.log(`"${indexName}" already exists`);
    }
}

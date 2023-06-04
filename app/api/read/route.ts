import { NextRequest, NextResponse } from 'next/server'
import { PineconeClient } from '@pinecone-database/pinecone'
import {
    queryPineconeVectorStoreAndQueryLLM,
} from '../../../utils'

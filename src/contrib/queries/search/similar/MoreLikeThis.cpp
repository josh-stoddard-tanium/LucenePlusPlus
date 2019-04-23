/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ContribInc.h"

#include "BooleanQuery.h"
#include "DefaultSimilarity.h"
#include "Document.h"
#include "MoreLikeThis.h"
#include "StringReader.h"
#include "Term.h"
#include "TermAttribute.h"
#include "TermFreqVector.h"
#include "TermQuery.h"
#include "TokenStream.h"

namespace Lucene
{

    const int32_t MoreLikeThis::DEFAULT_MAX_NUM_TOKENS_PARSED = 5000;
    const int32_t MoreLikeThis::DEFAULT_MIN_TERM_FREQ = 2;
    const int32_t MoreLikeThis::DEFAULT_MIN_DOC_FREQ = 5;
    const int32_t MoreLikeThis::DEFAULT_MAX_DOC_FREQ = std::numeric_limits< int32_t >::max();
    const bool MoreLikeThis::DEFAULT_BOOST = false;
    const int32_t MoreLikeThis::DEFAULT_MIN_WORD_LENGTH = 0;
    const int32_t MoreLikeThis::DEFAULT_MAX_WORD_LENGTH = 0;
    const int32_t MoreLikeThis::DEFAULT_MAX_QUERY_TERMS = 25;

    MoreLikeThis::MoreLikeThis( IndexReaderPtr ir )
        : stopWords( HashSet< String >::newInstance() )
        , minTermFreq( DEFAULT_MIN_TERM_FREQ )
        , minDocFreq( DEFAULT_MIN_DOC_FREQ )
        , maxDocFreq( DEFAULT_MAX_DOC_FREQ )
        , hasBoost( DEFAULT_BOOST )
        , fieldNames( HashSet< String >::newInstance() )
        , maxNumTokensParsed( DEFAULT_MAX_NUM_TOKENS_PARSED )
        , minWordLen( DEFAULT_MIN_WORD_LENGTH )
        , maxWordLen( DEFAULT_MAX_WORD_LENGTH )
        , maxQueryTerms( DEFAULT_MAX_QUERY_TERMS )
        , similarity( newLucene< DefaultSimilarity >() )
        , reader( ir )
        , boostFactor( 1. )
    {
    }

    MoreLikeThis::MoreLikeThis( IndexReaderPtr ir, SimilarityPtr sim )
        : stopWords( HashSet< String >::newInstance() )
        , minTermFreq( DEFAULT_MIN_TERM_FREQ )
        , minDocFreq( DEFAULT_MIN_DOC_FREQ )
        , maxDocFreq( DEFAULT_MAX_DOC_FREQ )
        , hasBoost( DEFAULT_BOOST )
        , fieldNames( HashSet< String >::newInstance() )
        , maxNumTokensParsed( DEFAULT_MAX_NUM_TOKENS_PARSED )
        , minWordLen( DEFAULT_MIN_WORD_LENGTH )
        , maxWordLen( DEFAULT_MAX_WORD_LENGTH )
        , maxQueryTerms( DEFAULT_MAX_QUERY_TERMS )
        , similarity( sim )
        , reader( ir )
        , boostFactor( 1. )
    {
    }

    /**
     * Return a query that will return docs like the passed lucene document ID.
     *
     * @param docNum the documentID of the lucene doc to generate the 'More Like This" query for.
     * @return a query that will return docs like the passed lucene document ID.
     */
    QueryPtr MoreLikeThis::like( int32_t docNum )
    {
        if ( fieldNames.empty() )
        {
            // gather list of valid fields from lucene
            fieldNames = reader->getFieldNames( IndexReader::FIELD_OPTION_INDEXED );
        }

        return createQuery( retrieveTerms( docNum ) );
    }

    /**
     * Return a query that will return docs like the passed Readers.
     * This was added in order to treat multi-value fields.
     *
     * @return a query that will return docs like the passed Readers.
     */
    QueryPtr MoreLikeThis::like( const String& fieldName, const Collection< ReaderPtr >& readers )
    {
        return createQuery( retrieveTerms( fieldName, readers) );
    }
    /**
     * Create the More like query from a PriorityQueue
     */
    QueryPtr MoreLikeThis::createQuery( TermScoreQueuePtr q )
    {
        BooleanQueryPtr query = newLucene< BooleanQuery >();
        TermScorePtr scoreTerm;
        double bestScore = -1.;

        while ( ( scoreTerm = q->pop() ) )
        {
            QueryPtr tq = newLucene< TermQuery >( scoreTerm->term );

            if ( hasBoost )
            {
                if ( bestScore == -1. )
                {
                    bestScore = ( scoreTerm->score );
                }
                double myScore = ( scoreTerm->score );
                tq->setBoost( boostFactor * myScore / bestScore );
            }

            try
            {
                query->add( tq, BooleanClause::SHOULD );
            }
            catch ( TooManyClausesException& /*ignore*/ )
            {
            }
        }
        return query;
    }

    /**
     * Create a PriorityQueue from a word-&gt;tf map.
     *
     * @param perFieldTermFrequencies a per field map of words keyed on the word(String) with int32_t objects as the
     * values.
     */
    MoreLikeThis::TermScoreQueuePtr MoreLikeThis::createQueue( const HashMap< String, HashMap< String, int32_t > >& perFieldTermFrequencies )
    {
        // have collected all words in doc and their freqs
        int32_t numDocs = reader->numDocs();
        const int32_t limit = ( std::min )( maxQueryTerms, getTermsCount( perFieldTermFrequencies ) );
        TermScoreQueuePtr queue = newLucene< TermScoreQueue >( limit );  // will order words by score
        for ( HashMap< String, HashMap< String, int32_t > >::const_iterator it = perFieldTermFrequencies.begin();
              it != perFieldTermFrequencies.end(); ++it )
        {
            const HashMap< String, int32_t >& perWordTermFrequencies = it->second;
            const String& fieldName = it->first;

            for ( HashMap< String, int32_t >::const_iterator it = perWordTermFrequencies.begin();
                  it != perWordTermFrequencies.end(); ++it )
            {  // for every word
                const String& word = it->first;
                int32_t tf = it->second;  // term freq in the source doc
                if ( minTermFreq > 0 && tf < minTermFreq )
                {
                    continue;  // filter out words that don't occur enough times in the source
                }

                TermPtr term = newLucene< Term >( fieldName, word );
                int32_t docFreq = reader->docFreq( term );

                if ( minDocFreq > 0 && docFreq < minDocFreq )
                {
                    continue;  // filter out words that don't occur in enough docs
                }

                if ( docFreq > maxDocFreq )
                {
                    continue;  // filter out words that occur in too many docs
                }

                if ( docFreq == 0 )
                {
                    continue;  // index update problem?
                }

                double idf = similarity->idf( docFreq, numDocs );
                double score = tf * idf;

                if ( queue->size() < limit )
                {
                    // there is still space in the queue
                    TermScorePtr st = newLucene< TermScore >();
                    st->term = term;
                    st->score = score;
                    queue->add( st );
                }
                else
                {
                    TermScorePtr st = queue->top();
                    if ( st->score < score )
                    {  // update the smallest in the queue in place and update the queue.
                        st->term = term;
                        st->score = score;
                        queue->updateTop();
                    }
                }
            }
        }
        return queue;
    }

    int32_t MoreLikeThis::getTermsCount( const HashMap< String, HashMap< String, int32_t > >& perFieldTermFrequencies )
    {
        int32_t totalTermsCount = 0;
        for ( HashMap< String, HashMap< String, int32_t > >::const_iterator it = perFieldTermFrequencies.begin();
              it != perFieldTermFrequencies.end(); ++it )
        {
            totalTermsCount += it->second.size();
        }
        return totalTermsCount;
    }

    /**
     * Describe the parameters that control how the "more like this" query is formed.
     */
    String MoreLikeThis::describeParams()
    {
        std::wstringstream ss;
        ss << L"\tmaxQueryTerms  : " << maxQueryTerms << std::endl;
        ss << L"\tminWordLen     : " << minWordLen << std::endl;
        ss << L"\tmaxWordLen     : " << maxWordLen << std::endl;
        ss << L"\tfieldNames     : ";
        String delim = L"";
        for ( HashSet< String >::const_iterator it = fieldNames.begin(); it != fieldNames.end(); ++it )
        {
            ss << delim << *it;
            delim = L", ";
        }
        ss << std::endl;
        ss << L"\tboost          : " << hasBoost << std::endl;
        ss << L"\tminTermFreq    : " << minTermFreq << std::endl;
        ss << L"\tminDocFreq     : " << minDocFreq << std::endl;
        return ss.str();
    }

    /**
     * Find words for a more-like-this query former.
     *
     * @param docNum the id of the lucene document from which to find terms
     */
    MoreLikeThis::TermScoreQueuePtr MoreLikeThis::retrieveTerms( int32_t docNum )
    {
        HashMap< String, HashMap< String, int32_t > > field2termFreqMap = HashMap< String, HashMap< String, int32_t > >::newInstance();
        for ( HashSet< String >::const_iterator it = fieldNames.begin(); it != fieldNames.end(); ++it )
        {
            const String& fieldName = *it;
            TermFreqVectorPtr vector = reader->getTermFreqVector( docNum, fieldName );

            // field does not store term vector info
            if ( !vector )
            {
                DocumentPtr d = reader->document( docNum );
                Collection< String > values = d->getValues( fieldName );
                for ( Collection< String >::const_iterator it = values.begin(); it != values.end(); ++it )
                {
                    addTermFrequencies( newLucene< StringReader >( *it ), field2termFreqMap, fieldName );
                }
            }
            else
            {
                addTermFrequencies( field2termFreqMap, vector, fieldName );
            }
        }

        return createQueue( field2termFreqMap );
    }

    /**
     * Find words for a more-like-this query former.
     *
     * @param docNum the id of the lucene document from which to find terms
     */
    MoreLikeThis::TermScoreQueuePtr MoreLikeThis::retrieveTerms( const String& fieldName, const Collection< ReaderPtr >& readers )
    {
        HashMap< String, HashMap< String, int32_t > > perFieldTermFrequencies =
            HashMap< String, HashMap< String, int32_t > >::newInstance();
        for ( Collection< ReaderPtr >::const_iterator it = readers.begin(); it != readers.end();
              ++it )
        {
            addTermFrequencies( *it, perFieldTermFrequencies, fieldName );
        }
        return createQueue( perFieldTermFrequencies );
    }

    /**
     * Adds terms and frequencies found in vector into the Map termFreqMap
     *
     * @param field2termFreqMap a Map of terms and their frequencies per field
     * @param vector List of terms and their frequencies for a doc/field
     */
    void MoreLikeThis::addTermFrequencies( HashMap< String, HashMap< String, int32_t > >& field2termFreqMap,
                             const TermFreqVectorPtr& vector,
                             const String& fieldName )
    {
        if ( vector )
        {
            if (!field2termFreqMap.contains( fieldName ) )
            {
                field2termFreqMap[fieldName] = HashMap< String, int32_t >::newInstance();
            }
            HashMap< String, int32_t >& termFreqMap = field2termFreqMap[fieldName];

            Collection< String > terms = vector->getTerms();
            Collection< int32_t > freqs = vector->getTermFrequencies();
            for ( int32_t ii = 0; ii < vector->size(); ++ii )
            {
                const String& term = terms[ii];
                if ( isNoiseWord( term ) )
                {
                    continue;
                }

                // increment frequency
                if ( !termFreqMap.contains( term ) )
                {
                    termFreqMap[term] = 0;
                }
                termFreqMap[term] += freqs[ii];
            }
        }
    }

    /**
     * Adds term frequencies found by tokenizing text from reader into the Map words
     *
     * @param r a source of text to be tokenized
     * @param perFieldTermFrequencies a Map of terms and their frequencies per field
     * @param fieldName Used by analyzer for any special per-field analysis
     */
    void MoreLikeThis::addTermFrequencies( const ReaderPtr& r,
                                           HashMap< String, HashMap< String, int32_t > >& perFieldTermFrequencies,
                                           const String& fieldName )
    {
        if ( !analyzer )
        {
            boost::throw_exception( UnsupportedOperationException( L"To use MoreLikeThis without term vectors, you must provide an Analyzer" ) );
        }

        if (!perFieldTermFrequencies.contains( fieldName ) )
        {
            perFieldTermFrequencies[fieldName] = HashMap< String, int32_t >::newInstance();
        }
        HashMap< String, int32_t >& termFreqMap = perFieldTermFrequencies[fieldName];

        TokenStreamPtr ts = analyzer->tokenStream( fieldName, r );
        int32_t tokenCount = 0;
        // for every token
        TermAttributePtr termAtt = ts->addAttribute< TermAttribute >();
        ts->reset();
        while ( ts->incrementToken() )
        {
            String term = termAtt->term();
            tokenCount++;
            if ( tokenCount > maxNumTokensParsed )
            {
                break;
            }
            if ( isNoiseWord( term ) )
            {
                continue;
            }

            // increment frequency
            if ( !termFreqMap.contains( term ) )
            {
                termFreqMap[term] = 0;
            }
            ++termFreqMap[term];
        }
        ts->end();
    }

    /**
     * determines if the passed term is likely to be of interest in "more like" comparisons
     *
     * @param term The word being considered
     * @return true if should be ignored, false if should be used in further analysis
     */
    bool MoreLikeThis::isNoiseWord( const String& term )
    {
        int32_t len = term.length();
        if ( minWordLen > 0 && len < minWordLen )
        {
            return true;
        }
        if ( maxWordLen > 0 && len > maxWordLen )
        {
            return true;
        }
        return stopWords.contains( term );
    }

    /**
     * Find words for a more-like-this query former.
     * The result is a priority queue of arrays with one entry for <b>every word</b> in the document.
     * Each array has 6 elements.
     * The elements are:
     * <ol>
     * <li> The word (String)
     * <li> The top field that this word comes from (String)
     * <li> The score for this word (Float)
     * <li> The IDF value (Float)
     * <li> The frequency of this word in the index (Integer)
     * <li> The frequency of this word in the source document (Integer)
     * </ol>
     * This is a somewhat "advanced" routine, and in general only the 1st entry in the array is of interest.
     * This method is exposed so that you can identify the "interesting words" in a document.
     * For an easier method to call see {@link #retrieveInterestingTerms retrieveInterestingTerms()}.
     *
     * @param r the reader that has the content of the document
     * @param fieldName field passed to the analyzer to use when analyzing the content
     * @return the most interesting words in the document ordered by score, with the highest scoring, or best entry,
     * first
     * @see #retrieveInterestingTerms
     */
    MoreLikeThis::TermScoreQueuePtr MoreLikeThis::retrieveTerms( const ReaderPtr& r, const String& fieldName )
    {
        HashMap< String, HashMap< String, int32_t > > field2termFreqMap = HashMap< String, HashMap< String, int32_t > >::newInstance();
        addTermFrequencies( r, field2termFreqMap, fieldName );
        return createQueue( field2termFreqMap );
    }

    /**
     * @see #retrieveInterestingTerms(java.io.Reader, String)
     */
    Collection< String > MoreLikeThis::retrieveInterestingTerms( int32_t docNum )
    {
        return retrieveInterestingTerms( retrieveTerms( docNum ) );
    }

    /**
     * @see #retrieveInterestingTerms(java.io.Reader, String)
     */
    Collection< String > MoreLikeThis::retrieveInterestingTerms( const String& fieldName, const Collection< ReaderPtr >& readers )
    {
        return retrieveInterestingTerms( retrieveTerms( fieldName, readers ) );
    }

    /**
     * Convenience routine to make it easy to return the most interesting words in a document.
     * More advanced users will call {@link #retrieveTerms(Reader, String) retrieveTerms()} directly.
     *
     * @param r the source document
     * @param fieldName field passed to analyzer to use when analyzing the content
     * @return the most interesting words in the document
     * @see #retrieveTerms(java.io.Reader, String)
     * @see #setMaxQueryTerms
     */
    Collection< String > MoreLikeThis::retrieveInterestingTerms( const ReaderPtr& r, const String& fieldName )
    {
        return retrieveInterestingTerms( retrieveTerms( r, fieldName ) );
    }

    Collection< String > MoreLikeThis::retrieveInterestingTerms( const TermScoreQueuePtr& pq )
    {
        TermScorePtr scoreTerm;
        int32_t lim = maxQueryTerms;  // have to be careful, retrieveTerms returns all words but that's probably not
                                      // useful to our caller...
        // we just want to return the top words
        Collection< String > res = Collection< String >::newInstance();
        while ( ( scoreTerm = pq->pop() ) && lim-- > 0 )
        {
            res.add( scoreTerm->term->_text );  // the 1st entry is the interesting word
        }
        return res;
    }

    int32_t MoreLikeThis::TermScore::compareTo( const LuceneObjectPtr& other )
    {
        const TermScorePtr tsp = boost::dynamic_pointer_cast< TermScore >( other );
        if (!tsp)
        {
            return -1;
        }
        if ( this->score == tsp->score )
        {
            return tsp->term->compareTo( this->term );
        }
        else
        {
            return this->score < tsp->score ? -1 : ( this->score > tsp->score ? 1 : 0 );
        }
    }

    MoreLikeThis::TermScoreQueue::TermScoreQueue( int32_t size )
        : PriorityQueue< TermScorePtr >( size )
    {
    }

    bool MoreLikeThis::TermScoreQueue::lessThan( const MoreLikeThis::TermScorePtr& first, const MoreLikeThis::TermScorePtr& second )
    {
        return ( first->compareTo( second ) < 0 );
    }
}  // namespace Lucene

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

#include "Analyzer.h"
#include "IndexReader.h"
#include "PriorityQueue.h"
#include "Query.h"
#include "Similarity.h"

/**
 * Generate "more like this" similarity queries.
 * Based on this mail:
 * <pre><code>
 * Lucene does let you access the document frequency of terms, with IndexReader.docFreq().
 * Term frequencies can be computed by re-tokenizing the text, which, for a single document,
 * is usually fast enough.  But looking up the docFreq() of every term in the document is
 * probably too slow.
 *
 * You can use some heuristics to prune the set of terms, to avoid calling docFreq() too much,
 * or at all.  Since you're trying to maximize a tf*idf score, you're probably most interested
 * in terms with a high tf. Choosing a tf threshold even as low as two or three will radically
 * reduce the number of terms under consideration.  Another heuristic is that terms with a
 * high idf (i.e., a low df) tend to be longer.  So you could threshold the terms by the
 * number of characters, not selecting anything less than, e.g., six or seven characters.
 * With these sorts of heuristics you can usually find small set of, e.g., ten or fewer terms
 * that do a pretty good job of characterizing a document.
 *
 * It all depends on what you're trying to do.  If you're trying to eek out that last percent
 * of precision and recall regardless of computational difficulty so that you can win a TREC
 * competition, then the techniques I mention above are useless.  But if you're trying to
 * provide a "more like this" button on a search results page that does a decent job and has
 * good performance, such techniques might be useful.
 *
 * An efficient, effective "more-like-this" query generator would be a great contribution, if
 * anyone's interested.  I'd imagine that it would take a Reader or a String (the document's
 * text), analyzer Analyzer, and return a set of representative terms using heuristics like those
 * above.  The frequency and length thresholds could be parameters, etc.
 *
 * Doug
 * </code></pre>
 * <h3>Initial Usage</h3>
 * <p>
 * This class has lots of options to try to make it efficient and flexible.
 * The simplest possible usage is as follows. The bold
 * fragment is specific to this class.
 * <br>
 * <pre class="prettyprint">
 * IndexReader ir = ...
 * IndexSearcher is = ...
 *
 * MoreLikeThis mlt = new MoreLikeThis(ir);
 * Reader target = ... // orig source of doc you want to find similarities to
 * Query query = mlt.like( target);
 *
 * Hits hits = is.search(query);
 * // now the usual iteration thru 'hits' - the only thing to watch for is to make sure
 * //you ignore the doc if it matches your 'target' document, as it should be similar to itself
 *
 * </pre>
 * <p>
 * Thus you:
 * <ol>
 * <li> do your normal, Lucene setup for searching,
 * <li> create a MoreLikeThis,
 * <li> get the text of the doc you want to find similarities to
 * <li> then call one of the like() calls to generate a similarity query
 * <li> call the searcher to find the similar docs
 * </ol>
 * <br>
 * <h3>More Advanced Usage</h3>
 * <p>
 * You may want to use {@link #setFieldNames setFieldNames(...)} so you can examine
 * multiple fields (e.g. body and title) for similarity.
 * <p>
 * Depending on the size of your index and the size and makeup of your documents you
 * may want to call the other set methods to control how the similarity queries are
 * generated:
 * <ul>
 * <li> {@link #setMinTermFreq setMinTermFreq(...)}
 * <li> {@link #setMinDocFreq setMinDocFreq(...)}
 * <li> {@link #setMaxDocFreq setMaxDocFreq(...)}
 * <li> {@link #setMaxDocFreqPct setMaxDocFreqPct(...)}
 * <li> {@link #setMinWordLen setMinWordLen(...)}
 * <li> {@link #setMaxWordLen setMaxWordLen(...)}
 * <li> {@link #setMaxQueryTerms setMaxQueryTerms(...)}
 * <li> {@link #setMaxNumTokensParsed setMaxNumTokensParsed(...)}
 * <li> {@link #setStopWords setStopWord(...)}
 * </ul>
 * <br>
 * <hr>
 * <pre>
 * Changes: Mark Harwood 29/02/04
 * Some bugfixing, some refactoring, some optimisation.
 * - bugfix: retrieveTerms(int32_t docNum) was not working for indexes without a termvector -added missing code
 * - bugfix: No significant terms being created for fields with a termvector - because
 * was only counting one occurrence per term/field pair in calculations(ie not including frequency info from TermVector)
 * - refactor: moved common code into isNoiseWord()
 * - optimise: when no termvector support available - used maxNumTermsParsed to limit amount of tokenization
 * </pre>
 */

namespace Lucene
{

class MoreLikeThis
{
public:
    /**
     * Default maximum number of tokens to parse in each example doc field that is not stored with TermVector support.
     *
     * @see #getMaxNumTokensParsed
     */
    static const int32_t DEFAULT_MAX_NUM_TOKENS_PARSED;

    /**
     * Ignore terms with less than this frequency in the source doc.
     *
     * @see #getMinTermFreq
     * @see #setMinTermFreq
     */
    static const int32_t DEFAULT_MIN_TERM_FREQ;

    /**
     * Ignore words which do not occur in at least this many docs.
     *
     * @see #getMinDocFreq
     * @see #setMinDocFreq
     */
    static const int32_t DEFAULT_MIN_DOC_FREQ;

    /**
     * Ignore words which occur in more than this many docs.
     *
     * @see #getMaxDocFreq
     * @see #setMaxDocFreq
     * @see #setMaxDocFreqPct
     */
    static const int32_t DEFAULT_MAX_DOC_FREQ;

    /**
     * Boost terms in query based on score.
     *
     * @see #isBoost
     * @see #setBoost
     */
    static const bool DEFAULT_BOOST;

    /**
     * Ignore words less than this length or if 0 then this has no effect.
     *
     * @see #getMinWordLen
     * @see #setMinWordLen
     */
    static const int32_t DEFAULT_MIN_WORD_LENGTH;

    /**
     * Ignore words greater than this length or if 0 then this has no effect.
     *
     * @see #getMaxWordLen
     * @see #setMaxWordLen
     */
    static const int32_t DEFAULT_MAX_WORD_LENGTH;

    /**
     * Return a Query with no more than this many terms.
     *
     * @see BooleanQuery#getMaxClauseCount
     * @see #getMaxQueryTerms
     * @see #setMaxQueryTerms
     */
    static const int32_t DEFAULT_MAX_QUERY_TERMS;

private:
    /**
     * Current set of stop words.
     */
    HashSet< String > stopWords;

    /**
     * Analyzer that will be used to parse the doc.
     */
    AnalyzerPtr analyzer;

    /**
     * Ignore words less frequent that this.
     */
    int32_t minTermFreq; // = DEFAULT_MIN_TERM_FREQ;

    /**
     * Ignore words which do not occur in at least this many docs.
     */
    int32_t minDocFreq; // = DEFAULT_MIN_DOC_FREQ;

    /**
     * Ignore words which occur in more than this many docs.
     */
    int32_t maxDocFreq; // = DEFAULT_MAX_DOC_FREQ;

    /**
     * Should we apply a boost to the Query based on the scores?
     */
    bool hasBoost; // = DEFAULT_BOOST;

    /**
     * Field name we'll analyze.
     */
    HashSet< String > fieldNames;

    /**
     * The maximum number of tokens to parse in each example doc field that is not stored with TermVector support
     */
    int32_t maxNumTokensParsed; // = DEFAULT_MAX_NUM_TOKENS_PARSED;

    /**
     * Ignore words if less than this len.
     */
    int32_t minWordLen; // = DEFAULT_MIN_WORD_LENGTH;

    /**
     * Ignore words if greater than this len.
     */
    int32_t maxWordLen; // = DEFAULT_MAX_WORD_LENGTH;

    /**
     * Don't return a query longer than this.
     */
    int32_t maxQueryTerms; // = DEFAULT_MAX_QUERY_TERMS;

    /**
     * For idf() calculations.
     */
    SimilarityPtr similarity;  // = new DefaultSimilarity();

    /**
     * IndexReader to use
     */
    IndexReaderPtr reader;

    /**
     * Boost factor to use when boosting the terms
     */
    double boostFactor;

public:
    /**
     * Returns the boost factor used when boosting terms
     *
     * @return the boost factor used when boosting terms
     * @see #setBoostFactor(double)
     */
    inline double getBoostFactor() const { return boostFactor; }

    /**
     * Sets the boost factor to use when boosting terms
     *
     * @see #getBoostFactor()
     */
    inline void setBoostFactor( double bf ) { boostFactor = bf; }

    /**
     * Constructor requiring an IndexReader.
     */
    MoreLikeThis( IndexReaderPtr ir );

    MoreLikeThis( IndexReaderPtr ir, SimilarityPtr sim );

    SimilarityPtr getSimilarity() const { return similarity; }

    void setSimilarity( SimilarityPtr sim ) { similarity = sim; }

    /**
     * Returns an analyzer that will be used to parse source doc with. The default analyzer
     * is not set.
     *
     * @return the analyzer that will be used to parse source doc with.
     */
    inline AnalyzerPtr getAnalyzer() const { return analyzer; }

    /**
     * Sets the analyzer to use. An analyzer is not required for generating a query with the
     * {@link #like(int32_t)} method, all other 'like' methods require an analyzer.
     *
     * @param analyzer the analyzer to use to tokenize text.
     */
    inline void setAnalyzer( AnalyzerPtr azr ) { analyzer = azr; }

    /**
     * Returns the frequency below which terms will be ignored in the source doc. The default
     * frequency is the {@link #DEFAULT_MIN_TERM_FREQ}.
     *
     * @return the frequency below which terms will be ignored in the source doc.
     */
    inline int32_t getMinTermFreq() const { return minTermFreq; }

    /**
     * Sets the frequency below which terms will be ignored in the source doc.
     *
     * @param minTermFreq the frequency below which terms will be ignored in the source doc.
     */
    inline void setMinTermFreq( int32_t mtf ) { minTermFreq = mtf; }

    /**
     * Returns the frequency at which words will be ignored which do not occur in at least this
     * many docs. The default frequency is {@link #DEFAULT_MIN_DOC_FREQ}.
     *
     * @return the frequency at which words will be ignored which do not occur in at least this
     *         many docs.
     */
    inline int32_t getMinDocFreq() const { return minDocFreq; }

    /**
     * Sets the frequency at which words will be ignored which do not occur in at least this
     * many docs.
     *
     * @param minDocFreq the frequency at which words will be ignored which do not occur in at
     * least this many docs.
     */
    inline void setMinDocFreq( int32_t mdf ) { minDocFreq = mdf; }

    /**
     * Returns the maximum frequency in which words may still appear.
     * Words that appear in more than this many docs will be ignored. The default frequency is
     * {@link #DEFAULT_MAX_DOC_FREQ}.
     *
     * @return get the maximum frequency at which words are still allowed,
     *         words which occur in more docs than this are ignored.
     */
    inline int32_t getMaxDocFreq() const { return maxDocFreq; }

    /**
     * Set the maximum frequency in which words may still appear. Words that appear
     * in more than this many docs will be ignored.
     *
     * @param mdf the maximum count of documents that a term may appear
     * in to be still considered relevant
     */
    inline void setMaxDocFreq( int32_t mdf ) { maxDocFreq = mdf; }

    /**
     * Set the maximum percentage in which words may still appear. Words that appear
     * in more than this many percent of all docs will be ignored.
     *
     * This method calls {@link #setMaxDocFreq(int32_t)} internally (both conditions cannot
     * be used at the same time).
     *
     * @param maxPercentage the maximum percentage of documents (0-100) that a term may appear
     * in to be still considered relevant.
     */
    inline void setMaxDocFreqPct( int32_t maxPercentage )
    {
        setMaxDocFreq( maxPercentage * reader->numDocs() / 100 );
    }

    /**
     * Returns whether to boost terms in query based on "score" or not. The default is
     * {@link #DEFAULT_BOOST}.
     *
     * @return whether to boost terms in query based on "score" or not.
     * @see #setBoost
     */
    inline bool isBoost() { return hasBoost; }

    /**
     * Sets whether to boost terms in query based on "score" or not.
     *
     * @param boost true to boost terms in query based on "score", false otherwise.
     * @see #isBoost
     */
    inline void setBoost( bool boost ) { hasBoost = boost; }

    /**
     * Returns the field names that will be used when generating the 'More Like This' query.
     * The default field names that will be used is {@link #DEFAULT_FIELD_NAMES}.
     *
     * @return the field names that will be used when generating the 'More Like This' query.
     */
    inline HashSet< String > getFieldNames() const { return fieldNames; }

    /**
     * Sets the field names that will be used when generating the 'More Like This' query.
     * Set this to null for the field names to be determined at runtime from the IndexReader
     * provided in the constructor.
     *
     * @param fieldNames the field names that will be used when generating the 'More Like This'
     * query.
     */
    inline void setFieldNames( HashSet< String > names ) { fieldNames = names; }

    /**
     * Returns the minimum word length below which words will be ignored. Set this to 0 for no
     * minimum word length. The default is {@link #DEFAULT_MIN_WORD_LENGTH}.
     *
     * @return the minimum word length below which words will be ignored.
     */
    inline int32_t getMinWordLen() const { return minWordLen; }

    /**
     * Sets the minimum word length below which words will be ignored.
     *
     * @param minWordLen the minimum word length below which words will be ignored.
     */
    inline void setMinWordLen( int32_t mwl ) { minWordLen = mwl; }

    /**
     * Returns the maximum word length above which words will be ignored. Set this to 0 for no
     * maximum word length. The default is {@link #DEFAULT_MAX_WORD_LENGTH}.
     *
     * @return the maximum word length above which words will be ignored.
     */
    inline int32_t getMaxWordLen() const { return maxWordLen; }

    /**
     * Sets the maximum word length above which words will be ignored.
     *
     * @param maxWordLen the maximum word length above which words will be ignored.
     */
    inline void setMaxWordLen( int32_t mwl ) { maxWordLen = mwl; }

    /**
     * Set the set of stopwords.
     * Any word in this set is considered "uninteresting" and ignored.
     * Even if your Analyzer allows stopwords, you might want to tell the MoreLikeThis code to ignore them, as
     * for the purposes of document similarity it seems reasonable to assume that "a stop word is never interesting".
     *
     * @param stopWords set of stopwords, if null it means to allow stop words
     * @see #getStopWords
     */ 
    inline void setStopWords( HashSet< String > sw) {
        stopWords = sw;
    }

    /**
     * Get the current stop words being used.
     *
     * @see #setStopWords
     */
    inline HashSet< String > getStopWords() const { return stopWords; }

    /**
     * Returns the maximum number of query terms that will be included in any generated query.
     * The default is {@link #DEFAULT_MAX_QUERY_TERMS}.
     *
     * @return the maximum number of query terms that will be included in any generated query.
     */
    inline int32_t getMaxQueryTerms() const { return maxQueryTerms; }

    /**
     * Sets the maximum number of query terms that will be included in any generated query.
     *
     * @param maxQueryTerms the maximum number of query terms that will be included in any
     * generated query.
     */
    inline void setMaxQueryTerms( int32_t mqt ) { maxQueryTerms = mqt; }

    /**
     * @return The maximum number of tokens to parse in each example doc field that is not stored with TermVector
     * support
     * @see #DEFAULT_MAX_NUM_TOKENS_PARSED
     */
    inline int32_t getMaxNumTokensParsed() const { return maxNumTokensParsed; }

    /**
     * @param i The maximum number of tokens to parse in each example doc field that is not stored with TermVector
     * support
     */
    inline void setMaxNumTokensParsed( int32_t i ) { maxNumTokensParsed = i; }

    /**
     * Return a query that will return docs like the passed lucene document ID.
     *
     * @param docNum the documentID of the lucene doc to generate the 'More Like This" query for.
     * @return a query that will return docs like the passed lucene document ID.
     */
    QueryPtr like( int32_t docNum );

    /**
     * Return a query that will return docs like the passed Readers.
     * This was added in order to treat multi-value fields.
     *
     * @return a query that will return docs like the passed Readers.
     */
    QueryPtr like( const String& fieldName, const Collection< ReaderPtr >& readers );

    /**
     * Describe the parameters that control how the "more like this" query is formed.
     */
    String describeParams();

    /**
     * @see #retrieveInterestingTerms(java.io.Reader, String)
     */
    Collection< String > retrieveInterestingTerms( int32_t docNum );

    /**
     * @see #retrieveInterestingTerms(java.io.Reader, String)
     */
    Collection< String > retrieveInterestingTerms( const String& fieldName, const Collection< ReaderPtr >& readers );

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
    Collection< String > retrieveInterestingTerms( const ReaderPtr& r, const String& fieldName );

private:

    class TermScore : public LuceneObject
    {
    public:
        LUCENE_CLASS( TermScore );

    public:
        TermPtr term;
        double score;

    public:
        virtual int32_t compareTo( const LuceneObjectPtr& other );
    };
    typedef boost::shared_ptr< TermScore > TermScorePtr;

    class TermScoreQueue : public PriorityQueue< TermScorePtr >
    {
    public:
        TermScoreQueue( int32_t size );
        LUCENE_CLASS( TermScoreQueue );

    protected:
        virtual bool lessThan( const TermScorePtr& first, const TermScorePtr& second );
    };
    typedef boost::shared_ptr< TermScoreQueue > TermScoreQueuePtr;

    /**
     * Create the More like query from a PriorityQueue
     */
    QueryPtr createQuery( TermScoreQueuePtr q );

    /**
     * Create a PriorityQueue from a word-&gt;tf map.
     *
     * @param perFieldTermFrequencies a per field map of words keyed on the word(String) with int32_t objects as the
     * values.
     */
    TermScoreQueuePtr createQueue( const HashMap< String, HashMap< String, int32_t > >& perFieldTermFrequencies );

    int32_t getTermsCount( const HashMap< String, HashMap< String, int32_t > >& perFieldTermFrequencies );

    /**
     * Find words for a more-like-this query former.
     *
     * @param docNum the id of the lucene document from which to find terms
     */
    TermScoreQueuePtr retrieveTerms( int32_t docNum );

    /**
     * Find words for a more-like-this query former.
     *
     * @param docNum the id of the lucene document from which to find terms
     */
    TermScoreQueuePtr retrieveTerms( const String& fieldName, const Collection< ReaderPtr >& readers );

    /**
     * Adds terms and frequencies found in vector into the Map termFreqMap
     *
     * @param field2termFreqMap a Map of terms and their frequencies per field
     * @param vector List of terms and their frequencies for a doc/field
     */
    void addTermFrequencies( HashMap< String, HashMap< String, int32_t > >& field2termFreqMap,
                             const TermFreqVectorPtr& tfvp,
                             const String& fieldName );

    /**
     * Adds term frequencies found by tokenizing text from reader into the Map words
     *
     * @param r a source of text to be tokenized
     * @param perFieldTermFrequencies a Map of terms and their frequencies per field
     * @param fieldName Used by analyzer for any special per-field analysis
     */
    void addTermFrequencies( const ReaderPtr& r,
                             HashMap< String, HashMap< String, int32_t > >& perFieldTermFrequencies,
                             const String& fieldName );

    /**
     * determines if the passed term is likely to be of interest in "more like" comparisons
     *
     * @param term The word being considered
     * @return true if should be ignored, false if should be used in further analysis
     */
    bool isNoiseWord( const String& term );

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
    TermScoreQueuePtr retrieveTerms( const ReaderPtr& r, const String& fieldName );

    Collection< String > retrieveInterestingTerms( const TermScoreQueuePtr& stq );
};
}

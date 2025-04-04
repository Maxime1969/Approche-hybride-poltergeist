package org.apache.lucene.search.spell;

/**
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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import org.apache.lucene.analysis.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.AlreadyClosedException;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.ReaderUtil;
import org.apache.lucene.util.Version;

/**
 * <p>
 *   Spell Checker class  (Main class) <br/>
 *  (initially inspired by the David Spencer code).
 * </p>
 *
 * <p>Example Usage:
 * 
 * <pre>
 *  SpellChecker spellchecker = new SpellChecker(spellIndexDirectory);
 *  // To index a field of a user index:
 *  spellchecker.indexDictionary(new LuceneDictionary(my_lucene_reader, a_field));
 *  // To index a file containing words:
 *  spellchecker.indexDictionary(new PlainTextDictionary(new File("myfile.txt")));
 *  String[] suggestions = spellchecker.suggestSimilar("misspelt", 5);
 * </pre>
 * 
 *
 * @version 1.0
 */
public class SpellChecker implements java.io.Closeable {

  /**
   * The default minimum score to use, if not specified by calling {@link #setAccuracy(float)} .
   */
  public static final float DEFAULT_ACCURACY = 0.5f;

  /**
   * Field name for each word in the ngram index.
   */
  public static final String F_WORD = "word";

  private static final Term F_WORD_TERM = new Term(F_WORD);

  /**
   * the spell index
   */
  // don't modify the directory directly - see #swapSearcher()
  // TODO: why is this package private?
  Directory spellIndex;
  /**
   * Boost value for start and end grams
   */
  private float bStart = 2.0f;

  private float bEnd = 1.0f;
  // don't use this searcher directly - see #swapSearcher()

  private IndexSearcher searcher;
  /*
   * this locks all modifications to the current searcher.
   */

  private final Object searcherLock = new Object();
  /*
   * this lock synchronizes all possible modifications to the
   * current index directory. It should not be possible to try modifying
   * the same index concurrently. Note: Do not acquire the searcher lock
   * before acquiring this lock!
   */
  private final Object modifyCurrentIndexLock = new Object();

  private volatile boolean closed = false;
  // minimum score for hits generated by the spell checker query

  private float accuracy = DEFAULT_ACCURACY;

  private StringDistance sd;
  private Comparator<SuggestWord> comparator;

  /**
   * Use the given directory as a spell checker index. The directory
   * is created if it doesn't exist yet.
   * @param spellIndex the spell index directory
   * @param sd the {@link StringDistance} measurement to use 
   * @throws IOException if Spellchecker can not open the directory
   */
  public SpellChecker(Directory spellIndex, StringDistance sd) throws IOException {
    this(spellIndex, sd, SuggestWordQueue.DEFAULT_COMPARATOR);
  }
  /**
   * Use the given directory as a spell checker index with a
   * {@link LevensteinDistance} as the default {@link StringDistance}. The
   * directory is created if it doesn't exist yet.
   * 
   * @param spellIndex
   *          the spell index directory
   * @throws IOException
   *           if spellchecker can not open the directory
   */
  public SpellChecker(Directory spellIndex) throws IOException {
    this(spellIndex, new LevensteinDistance());
  }

  /**
   * Use the given directory as a spell checker index with the given {@link org.apache.lucene.search.spell.StringDistance} measure
   * and the given {@link java.util.Comparator} for sorting the results.
   * @param spellIndex The spelling index
   * @param sd The distance
   * @param comparator The comparator
   * @throws IOException if there is a problem opening the index
   */
  public SpellChecker(Directory spellIndex, StringDistance sd, Comparator<SuggestWord> comparator) throws IOException {
    setSpellIndex(spellIndex);
    setStringDistance(sd);
    this.comparator = comparator;
  }
  
  /**
   * Use a different index as the spell checker index or re-open
   * the existing index if <code>spellIndex</code> is the same value
   * as given in the constructor.
   * @param spellIndexDir the spell directory to use
   * @throws AlreadyClosedException if the Spellchecker is already closed
   * @throws  IOException if spellchecker can not open the directory
   */
  // TODO: we should make this final as it is called in the constructor
  public void setSpellIndex(Directory spellIndexDir) throws IOException {
    // this could be the same directory as the current spellIndex
    // modifications to the directory should be synchronized 
    synchronized (modifyCurrentIndexLock) {
      ensureOpen();
      if (!IndexReader.indexExists(spellIndexDir)) {
          IndexWriter writer = new IndexWriter(spellIndexDir,
            new IndexWriterConfig(Version.LUCENE_CURRENT,
                new WhitespaceAnalyzer(Version.LUCENE_CURRENT)));
          writer.close();
      }
      swapSearcher(spellIndexDir);
    }
  }

  /**
   * Sets the {@link java.util.Comparator} for the {@link SuggestWordQueue}.
   * @param comparator the comparator
   */
  public void setComparator(Comparator<SuggestWord> comparator) {
    this.comparator = comparator;
  }

  public Comparator<SuggestWord> getComparator() {
    return comparator;
  }

  /**
   * Sets the {@link StringDistance} implementation for this
   * {@link SpellChecker} instance.
   * 
   * @param sd the {@link StringDistance} implementation for this
   * {@link SpellChecker} instance
   */
  public void setStringDistance(StringDistance sd) {
    this.sd = sd;
  }
  /**
   * Returns the {@link StringDistance} instance used by this
   * {@link SpellChecker} instance.
   * 
   * @return the {@link StringDistance} instance used by this
   *         {@link SpellChecker} instance.
   */
  public StringDistance getStringDistance() {
    return sd;
  }

  /**
   * Sets the accuracy 0 &lt; minScore &lt; 1; default {@link #DEFAULT_ACCURACY}
   * @param acc The new accuracy
   */
  public void setAccuracy(float acc) {
    this.accuracy = acc;
  }

  /**
   * The accuracy (minimum score) to be used, unless overridden in {@link #suggestSimilar(String, int, org.apache.lucene.index.IndexReader, String, boolean, float)}, to
   * decide whether a suggestion is included or not.
   * @return The current accuracy setting
   */
  public float getAccuracy() {
    return accuracy;
  }

  /**
   * Suggest similar words.
   * 
   * <p>As the Lucene similarity that is used to fetch the most relevant n-grammed terms
   * is not the same as the edit distance strategy used to calculate the best
   * matching spell-checked word from the hits that Lucene found, one usually has
   * to retrieve a couple of numSug's in order to get the true best match.
   *
   * <p>I.e. if numSug == 1, don't count on that suggestion being the best one.
   * Thus, you should set this value to <b>at least</b> 5 for a good suggestion.
   *
   * @param word the word you want a spell check done on
   * @param numSug the number of suggested words
   * @throws IOException if the underlying index throws an {@link IOException}
   * @throws AlreadyClosedException if the Spellchecker is already closed
   * @return String[]
   *
   * @see #suggestSimilar(String, int, org.apache.lucene.index.IndexReader, String, boolean, float) 
   */
  public String[] suggestSimilar(String word, int numSug) throws IOException {
    return this.suggestSimilar(word, numSug, null, null, false);
  }

  /**
   * Suggest similar words.
   *
   * <p>As the Lucene similarity that is used to fetch the most relevant n-grammed terms
   * is not the same as the edit distance strategy used to calculate the best
   * matching spell-checked word from the hits that Lucene found, one usually has
   * to retrieve a couple of numSug's in order to get the true best match.
   *
   * <p>I.e. if numSug == 1, don't count on that suggestion being the best one.
   * Thus, you should set this value to <b>at least</b> 5 for a good suggestion.
   *
   * @param word the word you want a spell check done on
   * @param numSug the number of suggested words
   * @param accuracy The minimum score a suggestion must have in order to qualify for inclusion in the results
   * @throws IOException if the underlying index throws an {@link IOException}
   * @throws AlreadyClosedException if the Spellchecker is already closed
   * @return String[]
   *
   * @see #suggestSimilar(String, int, org.apache.lucene.index.IndexReader, String, boolean, float)
   */
  public String[] suggestSimilar(String word, int numSug, float accuracy) throws IOException {
    return this.suggestSimilar(word, numSug, null, null, false, accuracy);
  }

  /**
   * Suggest similar words (optionally restricted to a field of an index).
   * 
   * <p>As the Lucene similarity that is used to fetch the most relevant n-grammed terms
   * is not the same as the edit distance strategy used to calculate the best
   * matching spell-checked word from the hits that Lucene found, one usually has
   * to retrieve a couple of numSug's in order to get the true best match.
   *
   * <p>I.e. if numSug == 1, don't count on that suggestion being the best one.
   * Thus, you should set this value to <b>at least</b> 5 for a good suggestion.
   *
   * <p>Uses the {@link #getAccuracy()} value passed into the constructor as the accuracy.
   *
   * @param word the word you want a spell check done on
   * @param numSug the number of suggested words
   * @param ir the indexReader of the user index (can be null see field param)
   * @param field the field of the user index: if field is not null, the suggested
   * words are restricted to the words present in this field.
   * @param morePopular return only the suggest words that are as frequent or more frequent than the searched word
   * (only if restricted mode = (indexReader!=null and field!=null)
   * @throws IOException if the underlying index throws an {@link IOException}
   * @throws AlreadyClosedException if the Spellchecker is already closed
   * @return String[] the sorted list of the suggest words with these 2 criteria:
   * first criteria: the edit distance, second criteria (only if restricted mode): the popularity
   * of the suggest words in the field of the user index
   *
   * @see #suggestSimilar(String, int, org.apache.lucene.index.IndexReader, String, boolean, float)
   */
  public String[] suggestSimilar(String word, int numSug, IndexReader ir,
      String field, boolean morePopular) throws IOException {
    return suggestSimilar(word, numSug, ir, field, morePopular, accuracy);
  }


  /**
   * Suggest similar words (optionally restricted to a field of an index).
   *
   * <p>As the Lucene similarity that is used to fetch the most relevant n-grammed terms
   * is not the same as the edit distance strategy used to calculate the best
   * matching spell-checked word from the hits that Lucene found, one usually has
   * to retrieve a couple of numSug's in order to get the true best match.
   *
   * <p>I.e. if numSug == 1, don't count on that suggestion being the best one.
   * Thus, you should set this value to <b>at least</b> 5 for a good suggestion.
   *
   * @param word the word you want a spell check done on
   * @param numSug the number of suggested words
   * @param ir the indexReader of the user index (can be null see field param)
   * @param field the field of the user index: if field is not null, the suggested
   * words are restricted to the words present in this field.
   * @param morePopular return only the suggest words that are as frequent or more frequent than the searched word
   * (only if restricted mode = (indexReader!=null and field!=null)
   * @param accuracy The minimum score a suggestion must have in order to qualify for inclusion in the results
   * @throws IOException if the underlying index throws an {@link IOException}
   * @throws AlreadyClosedException if the Spellchecker is already closed
   * @return String[] the sorted list of the suggest words with these 2 criteria:
   * first criteria: the edit distance, second criteria (only if restricted mode): the popularity
   * of the suggest words in the field of the user index
   */
  public String[] suggestSimilar(String word, int numSug, IndexReader ir,
      String field, boolean morePopular, float accuracy) throws IOException {
    // obtainSearcher calls ensureOpen
    final IndexSearcher indexSearcher = obtainSearcher();
    try{

      final int lengthWord = word.length();

      final int freq = (ir != null && field != null) ? ir.docFreq(new Term(field, word)) : 0;
      final int goalFreq = (morePopular && ir != null && field != null) ? freq : 0;
      // if the word exists in the real index and we don't care for word frequency, return the word itself
      if (!morePopular && freq > 0) {
        return new String[] { word };
      }

      BooleanQuery query = new BooleanQuery();
      String[] grams;
      String key;

      for (int ng = getMin(lengthWord); ng <= getMax(lengthWord); ng++) {

        key = "gram" + ng; // form key

        grams = formGrams(word, ng); // form word into ngrams (allow dups too)

        if (grams.length == 0) {
          continue; // hmm
        }

        if (bStart > 0) { // should we boost prefixes?
          add(query, "start" + ng, grams[0], bStart); // matches start of word

        }
        if (bEnd > 0) { // should we boost suffixes
          add(query, "end" + ng, grams[grams.length - 1], bEnd); // matches end of word

        }
        for (int i = 0; i < grams.length; i++) {
          add(query, key, grams[i]);
        }
      }

      int maxHits = 10 * numSug;

  //    System.out.println("Q: " + query);
      ScoreDoc[] hits = indexSearcher.search(query, null, maxHits).scoreDocs;
  //    System.out.println("HITS: " + hits.length());
      SuggestWordQueue sugQueue = new SuggestWordQueue(numSug, comparator);

      // go thru more than 'maxr' matches in case the distance filter triggers
      int stop = Math.min(hits.length, maxHits);
      SuggestWord sugWord = new SuggestWord();
      for (int i = 0; i < stop; i++) {

        sugWord.string = indexSearcher.doc(hits[i].doc).get(F_WORD); // get orig word

        // don't suggest a word for itself, that would be silly
        if (sugWord.string.equals(word)) {
          continue;
        }

        // edit distance
        sugWord.score = sd.getDistance(word,sugWord.string);
        if (sugWord.score < accuracy) {
          continue;
        }

        if (ir != null && field != null) { // use the user index
          sugWord.freq = ir.docFreq(new Term(field, sugWord.string)); // freq in the index
          // don't suggest a word that is not present in the field
          if ((morePopular && goalFreq > sugWord.freq) || sugWord.freq < 1) {
            continue;
          }
        }
        sugQueue.insertWithOverflow(sugWord);
        if (sugQueue.size() == numSug) {
          // if queue full, maintain the minScore score
          accuracy = sugQueue.top().score;
        }
        sugWord = new SuggestWord();
      }

      // convert to array string
      String[] list = new String[sugQueue.size()];
      for (int i = sugQueue.size() - 1; i >= 0; i--) {
        list[i] = sugQueue.pop().string;
      }

      return list;
    } finally {
      releaseSearcher(indexSearcher);
    }
  }
  /**
   * Add a clause to a boolean query.
   */
  private static void add(BooleanQuery q, String name, String value, float boost) {
    Query tq = new TermQuery(new Term(name, value));
    tq.setBoost(boost);
    q.add(new BooleanClause(tq, BooleanClause.Occur.SHOULD));
  }

  /**
   * Add a clause to a boolean query.
   */
  private static void add(BooleanQuery q, String name, String value) {
    q.add(new BooleanClause(new TermQuery(new Term(name, value)), BooleanClause.Occur.SHOULD));
  }

  /**
   * Form all ngrams for a given word.
   * @param text the word to parse
   * @param ng the ngram length e.g. 3
   * @return an array of all ngrams in the word and note that duplicates are not removed
   */
  private static String[] formGrams(String text, int ng) {
    int len = text.length();
    String[] res = new String[len - ng + 1];
    for (int i = 0; i < len - ng + 1; i++) {
      res[i] = text.substring(i, i + ng);
    }
    return res;
  }

  /**
   * Removes all terms from the spell check index.
   * @throws IOException
   * @throws AlreadyClosedException if the Spellchecker is already closed
   */
  public void clearIndex() throws IOException {
    synchronized (modifyCurrentIndexLock) {
      ensureOpen();
      final Directory dir = this.spellIndex;
      final IndexWriter writer = new IndexWriter(dir, new IndexWriterConfig(
          Version.LUCENE_CURRENT,
          new WhitespaceAnalyzer(Version.LUCENE_CURRENT))
          .setOpenMode(OpenMode.CREATE));
      writer.close();
      swapSearcher(dir);
    }
  }

  /**
   * Check whether the word exists in the index.
   * @param word
   * @throws IOException
   * @throws AlreadyClosedException if the Spellchecker is already closed
   * @return true if the word exists in the index
   */
  public boolean exist(String word) throws IOException {
    // obtainSearcher calls ensureOpen
    final IndexSearcher indexSearcher = obtainSearcher();
    try{
      return indexSearcher.docFreq(F_WORD_TERM.createTerm(word)) > 0;
    } finally {
      releaseSearcher(indexSearcher);
    }
  }

  /**
   * Indexes the data from the given {@link Dictionary}.
   * @param dict Dictionary to index
   * @param mergeFactor mergeFactor to use when indexing
   * @param ramMB the max amount or memory in MB to use
   * @param optimize whether or not the spellcheck index should be optimized
   * @throws AlreadyClosedException if the Spellchecker is already closed
   * @throws IOException
   */
  public final void indexDictionary(Dictionary dict, int mergeFactor, int ramMB, boolean optimize) throws IOException {
    synchronized (modifyCurrentIndexLock) {
      ensureOpen();
      final Directory dir = this.spellIndex;
      final IndexWriter writer = new IndexWriter(dir, new IndexWriterConfig(Version.LUCENE_CURRENT, new WhitespaceAnalyzer(Version.LUCENE_CURRENT)).setRAMBufferSizeMB(ramMB));
      ((TieredMergePolicy) writer.getConfig().getMergePolicy()).setMaxMergeAtOnce(mergeFactor);
      IndexSearcher indexSearcher = obtainSearcher();
      final List<IndexReader> readers = new ArrayList<IndexReader>();

      if (searcher.maxDoc() > 0) {
        ReaderUtil.gatherSubReaders(readers, searcher.getIndexReader());
      }
      
      boolean isEmpty = readers.isEmpty();

      try { 
        Iterator<String> iter = dict.getWordsIterator();
        
        terms: while (iter.hasNext()) {
          String word = iter.next();
  
          int len = word.length();
          if (len < 3) {
            continue; // too short we bail but "too long" is fine...
          }
  
          if (!isEmpty) {
            // we have a non-empty index, check if the term exists
            Term term = F_WORD_TERM.createTerm(word);
            for (IndexReader ir : readers) {
              if (ir.docFreq(term) > 0) {
                continue terms;
              }
            }
          }
  
          // ok index the word
          Document doc = createDocument(word, getMin(len), getMax(len));
          writer.addDocument(doc);
        }
      } finally {
        releaseSearcher(indexSearcher);
      }
      // close writer
      if (optimize)
        writer.optimize();
      writer.close();
      // also re-open the spell index to see our own changes when the next suggestion
      // is fetched:
      swapSearcher(dir);
    }
  }

  /**
   * Indexes the data from the given {@link Dictionary}.
   * @param dict the dictionary to index
   * @param mergeFactor mergeFactor to use when indexing
   * @param ramMB the max amount or memory in MB to use
   * @throws IOException
   */
  public final void indexDictionary(Dictionary dict, int mergeFactor, int ramMB) throws IOException {
    indexDictionary(dict, mergeFactor, ramMB, true);
  }
  
  /**
   * Indexes the data from the given {@link Dictionary}.
   * @param dict the dictionary to index
   * @throws IOException
   */
  public final void indexDictionary(Dictionary dict) throws IOException {
    indexDictionary(dict, 300, (int)IndexWriterConfig.DEFAULT_RAM_BUFFER_SIZE_MB);
  }

  private static int getMin(int l) {
    if (l > 5) {
      return 3;
    }
    if (l == 5) {
      return 2;
    }
    return 1;
  }

  private static int getMax(int l) {
    if (l > 5) {
      return 4;
    }
    if (l == 5) {
      return 3;
    }
    return 2;
  }

  private static Document createDocument(String text, int ng1, int ng2) {
    Document doc = new Document();
    // the word field is never queried on... its indexed so it can be quickly
    // checked for rebuild (and stored for retrieval). Doesn't need norms or TF/pos
    Field f = new Field(F_WORD, text, Field.Store.YES, Field.Index.NOT_ANALYZED);
    f.setOmitTermFreqAndPositions(true);
    f.setOmitNorms(true);
    doc.add(f); // orig term
    addGram(text, doc, ng1, ng2);
    return doc;
  }

  private static void addGram(String text, Document doc, int ng1, int ng2) {
    int len = text.length();
    for (int ng = ng1; ng <= ng2; ng++) {
      String key = "gram" + ng;
      String end = null;
      for (int i = 0; i < len - ng + 1; i++) {
        String gram = text.substring(i, i + ng);
        doc.add(new Field(key, gram, Field.Store.NO, Field.Index.NOT_ANALYZED));
        if (i == 0) {
          // only one term possible in the startXXField, TF/pos and norms aren't needed.
          Field startField = new Field("start" + ng, gram, Field.Store.NO, Field.Index.NOT_ANALYZED);
          startField.setOmitTermFreqAndPositions(true);
          startField.setOmitNorms(true);
          doc.add(startField);
        }
        end = gram;
      }
      if (end != null) { // may not be present if len==ng1
        // only one term possible in the endXXField, TF/pos and norms aren't needed.
        Field endField = new Field("end" + ng, end, Field.Store.NO, Field.Index.NOT_ANALYZED);
        endField.setOmitTermFreqAndPositions(true);
        endField.setOmitNorms(true);
        doc.add(endField);
      }
    }
  }
  
  private IndexSearcher obtainSearcher() {
    synchronized (searcherLock) {
      ensureOpen();
      searcher.getIndexReader().incRef();
      return searcher;
    }
  }
  
  private void releaseSearcher(final IndexSearcher aSearcher) throws IOException{
      // don't check if open - always decRef 
      // don't decrement the private searcher - could have been swapped
      aSearcher.getIndexReader().decRef();      
  }
  
  private void ensureOpen() {
    if (closed) {
      throw new AlreadyClosedException("Spellchecker has been closed");
    }
  }
  
  /**
   * Close the IndexSearcher used by this SpellChecker
   * @throws IOException if the close operation causes an {@link IOException}
   * @throws AlreadyClosedException if the {@link SpellChecker} is already closed
   */
  public void close() throws IOException {
    synchronized (searcherLock) {
      ensureOpen();
      closed = true;
      if (searcher != null) {
        searcher.close();
      }
      searcher = null;
    }
  }
  
  private void swapSearcher(final Directory dir) throws IOException {
    /*
     * opening a searcher is possibly very expensive.
     * We rather close it again if the Spellchecker was closed during
     * this operation than block access to the current searcher while opening.
     */
    final IndexSearcher indexSearcher = createSearcher(dir);
    synchronized (searcherLock) {
      if(closed){
        indexSearcher.close();
        throw new AlreadyClosedException("Spellchecker has been closed");
      }
      if (searcher != null) {
        searcher.close();
      }
      // set the spellindex in the sync block - ensure consistency.
      searcher = indexSearcher;
      this.spellIndex = dir;
    }
  }
  
  /**
   * Creates a new read-only IndexSearcher 
   * @param dir the directory used to open the searcher
   * @return a new read-only IndexSearcher
   * @throws IOException f there is a low-level IO error
   */
  // for testing purposes
  IndexSearcher createSearcher(final Directory dir) throws IOException{
    return new IndexSearcher(dir, true);
  }
  
  /**
   * Returns <code>true</code> if and only if the {@link SpellChecker} is
   * closed, otherwise <code>false</code>.
   * 
   * @return <code>true</code> if and only if the {@link SpellChecker} is
   *         closed, otherwise <code>false</code>.
   */
  boolean isClosed(){
    return closed;
  }
  
}

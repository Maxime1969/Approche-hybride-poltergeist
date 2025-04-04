package org.apache.lucene.search;

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

import java.io.Reader;
import java.io.IOException;
import java.io.StringReader;
import java.util.Collection;
import java.util.Collections;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.WhitespaceAnalyzer;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.RandomIndexWriter;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermPositions;
import org.apache.lucene.queryParser.QueryParser;
import org.apache.lucene.store.Directory;
import org.apache.lucene.analysis.LowerCaseTokenizer;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.index.Payload;
import org.apache.lucene.search.payloads.PayloadSpanUtil;
import org.apache.lucene.search.spans.SpanNearQuery;
import org.apache.lucene.search.spans.SpanQuery;
import org.apache.lucene.search.spans.SpanTermQuery;
import org.apache.lucene.search.spans.Spans;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.LuceneTestCase;

/**
 * Term position unit test.
 *
 *
 * @version $Revision: 1066722 $
 */
public class TestPositionIncrement extends LuceneTestCase {

  public void testSetPosition() throws Exception {
    Analyzer analyzer = new Analyzer() {
      @Override
      public TokenStream tokenStream(String fieldName, Reader reader) {
        return new TokenStream() {
          private final String[] TOKENS = {"1", "2", "3", "4", "5"};
          private final int[] INCREMENTS = {0, 2, 1, 0, 1};
          private int i = 0;

          PositionIncrementAttribute posIncrAtt = addAttribute(PositionIncrementAttribute.class);
          CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);
          OffsetAttribute offsetAtt = addAttribute(OffsetAttribute.class);
          
          @Override
          public boolean incrementToken() {
            if (i == TOKENS.length)
              return false;
            clearAttributes();
            termAtt.append(TOKENS[i]);
            offsetAtt.setOffset(i,i);
            posIncrAtt.setPositionIncrement(INCREMENTS[i]);
            i++;
            return true;
          }
        };
      }
    };
    Directory store = newDirectory();
    RandomIndexWriter writer = new RandomIndexWriter(random, store, analyzer);
    Document d = new Document();
    d.add(newField("field", "bogus", Field.Store.YES, Field.Index.ANALYZED));
    writer.addDocument(d);
    IndexReader reader = writer.getReader();
    writer.close();
    

    IndexSearcher searcher = newSearcher(reader);
    
    TermPositions pos = searcher.getIndexReader().termPositions(new Term("field", "1"));
    pos.next();
    // first token should be at position 0
    assertEquals(0, pos.nextPosition());
    
    pos = searcher.getIndexReader().termPositions(new Term("field", "2"));
    pos.next();
    // second token should be at position 2
    assertEquals(2, pos.nextPosition());
    
    PhraseQuery q;
    ScoreDoc[] hits;

    q = new PhraseQuery();
    q.add(new Term("field", "1"));
    q.add(new Term("field", "2"));
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(0, hits.length);

    // same as previous, just specify positions explicitely.
    q = new PhraseQuery(); 
    q.add(new Term("field", "1"),0);
    q.add(new Term("field", "2"),1);
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(0, hits.length);

    // specifying correct positions should find the phrase.
    q = new PhraseQuery();
    q.add(new Term("field", "1"),0);
    q.add(new Term("field", "2"),2);
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(1, hits.length);

    q = new PhraseQuery();
    q.add(new Term("field", "2"));
    q.add(new Term("field", "3"));
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(1, hits.length);

    q = new PhraseQuery();
    q.add(new Term("field", "3"));
    q.add(new Term("field", "4"));
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(0, hits.length);

    // phrase query would find it when correct positions are specified. 
    q = new PhraseQuery();
    q.add(new Term("field", "3"),0);
    q.add(new Term("field", "4"),0);
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(1, hits.length);

    // phrase query should fail for non existing searched term 
    // even if there exist another searched terms in the same searched position. 
    q = new PhraseQuery();
    q.add(new Term("field", "3"),0);
    q.add(new Term("field", "9"),0);
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(0, hits.length);

    // multi-phrase query should succed for non existing searched term
    // because there exist another searched terms in the same searched position. 
    MultiPhraseQuery mq = new MultiPhraseQuery();
    mq.add(new Term[]{new Term("field", "3"),new Term("field", "9")},0);
    hits = searcher.search(mq, null, 1000).scoreDocs;
    assertEquals(1, hits.length);

    q = new PhraseQuery();
    q.add(new Term("field", "2"));
    q.add(new Term("field", "4"));
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(1, hits.length);

    q = new PhraseQuery();
    q.add(new Term("field", "3"));
    q.add(new Term("field", "5"));
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(1, hits.length);

    q = new PhraseQuery();
    q.add(new Term("field", "4"));
    q.add(new Term("field", "5"));
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(1, hits.length);

    q = new PhraseQuery();
    q.add(new Term("field", "2"));
    q.add(new Term("field", "5"));
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(0, hits.length);

    // should not find "1 2" because there is a gap of 1 in the index
    QueryParser qp = new QueryParser(TEST_VERSION_CURRENT, "field",
                                     new StopWhitespaceAnalyzer(false));
    q = (PhraseQuery) qp.parse("\"1 2\"");
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(0, hits.length);

    // omitted stop word cannot help because stop filter swallows the increments. 
    q = (PhraseQuery) qp.parse("\"1 stop 2\"");
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(0, hits.length);

    // query parser alone won't help, because stop filter swallows the increments. 
    qp.setEnablePositionIncrements(true);
    q = (PhraseQuery) qp.parse("\"1 stop 2\"");
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(0, hits.length);

    // stop filter alone won't help, because query parser swallows the increments. 
    qp.setEnablePositionIncrements(false);
    q = (PhraseQuery) qp.parse("\"1 stop 2\"");
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(0, hits.length);
      
    // when both qp qnd stopFilter propagate increments, we should find the doc.
    qp = new QueryParser(TEST_VERSION_CURRENT, "field",
                         new StopWhitespaceAnalyzer(true));
    qp.setEnablePositionIncrements(true);
    q = (PhraseQuery) qp.parse("\"1 stop 2\"");
    hits = searcher.search(q, null, 1000).scoreDocs;
    assertEquals(1, hits.length);
    
    searcher.close();
    reader.close();
    store.close();
  }

  private static class StopWhitespaceAnalyzer extends Analyzer {
    boolean enablePositionIncrements;
    final WhitespaceAnalyzer a = new WhitespaceAnalyzer(TEST_VERSION_CURRENT);
    public StopWhitespaceAnalyzer(boolean enablePositionIncrements) {
      this.enablePositionIncrements = enablePositionIncrements;
    }
    @Override
    public TokenStream tokenStream(String fieldName, Reader reader) {
      TokenStream ts = a.tokenStream(fieldName,reader);
      return new StopFilter(enablePositionIncrements?TEST_VERSION_CURRENT:Version.LUCENE_24, ts,
          new CharArraySet(TEST_VERSION_CURRENT, Collections.singleton("stop"), true));
    }
  }
  
  public void testPayloadsPos0() throws Exception {
    Directory dir = newDirectory();
    RandomIndexWriter writer = new RandomIndexWriter(random, dir, new TestPayloadAnalyzer());
    Document doc = new Document();
    doc.add(new Field("content",
                      new StringReader("a a b c d e a f g h i j a b k k")));
    writer.addDocument(doc);

    IndexReader r = writer.getReader();

    TermPositions tp = r.termPositions(new Term("content", "a"));
    int count = 0;
    assertTrue(tp.next());
    // "a" occurs 4 times
    assertEquals(4, tp.freq());
    int expected = 0;
    assertEquals(expected, tp.nextPosition());
    assertEquals(1, tp.nextPosition());
    assertEquals(3, tp.nextPosition());
    assertEquals(6, tp.nextPosition());

    // only one doc has "a"
    assertFalse(tp.next());

    IndexSearcher is = newSearcher(r);
  
    SpanTermQuery stq1 = new SpanTermQuery(new Term("content", "a"));
    SpanTermQuery stq2 = new SpanTermQuery(new Term("content", "k"));
    SpanQuery[] sqs = { stq1, stq2 };
    SpanNearQuery snq = new SpanNearQuery(sqs, 30, false);

    count = 0;
    boolean sawZero = false;
    //System.out.println("\ngetPayloadSpans test");
    Spans pspans = snq.getSpans(is.getIndexReader());
    while (pspans.next()) {
      //System.out.println(pspans.doc() + " - " + pspans.start() + " - "+ pspans.end());
      Collection<byte[]> payloads = pspans.getPayload();
      sawZero |= pspans.start() == 0;
      count += payloads.size();
    }
    assertEquals(5, count);
    assertTrue(sawZero);

    //System.out.println("\ngetSpans test");
    Spans spans = snq.getSpans(is.getIndexReader());
    count = 0;
    sawZero = false;
    while (spans.next()) {
      count++;
      sawZero |= spans.start() == 0;
      //System.out.println(spans.doc() + " - " + spans.start() + " - " + spans.end());
    }
    assertEquals(4, count);
    assertTrue(sawZero);
  
    //System.out.println("\nPayloadSpanUtil test");

    sawZero = false;
    PayloadSpanUtil psu = new PayloadSpanUtil(is.getIndexReader());
    Collection<byte[]> pls = psu.getPayloadsForQuery(snq);
    count = pls.size();
    for (byte[] bytes : pls) {
      String s = new String(bytes);
      //System.out.println(s);
      sawZero |= s.equals("pos: 0");
    }
    assertEquals(5, count);
    assertTrue(sawZero);
    writer.close();
    is.getIndexReader().close();
    dir.close();
  }
}

final class TestPayloadAnalyzer extends Analyzer {

  @Override
  public TokenStream tokenStream(String fieldName, Reader reader) {
    TokenStream result = new LowerCaseTokenizer(LuceneTestCase.TEST_VERSION_CURRENT, reader);
    return new PayloadFilter(result, fieldName);
  }
}

final class PayloadFilter extends TokenFilter {
  String fieldName;

  int pos;

  int i;

  final PositionIncrementAttribute posIncrAttr;
  final PayloadAttribute payloadAttr;
  final CharTermAttribute termAttr;

  public PayloadFilter(TokenStream input, String fieldName) {
    super(input);
    this.fieldName = fieldName;
    pos = 0;
    i = 0;
    posIncrAttr = input.addAttribute(PositionIncrementAttribute.class);
    payloadAttr = input.addAttribute(PayloadAttribute.class);
    termAttr = input.addAttribute(CharTermAttribute.class);
  }

  @Override
  public boolean incrementToken() throws IOException {
    if (input.incrementToken()) {
      payloadAttr.setPayload(new Payload(("pos: " + pos).getBytes()));
      int posIncr;
      if (i % 2 == 1) {
        posIncr = 1;
      } else {
        posIncr = 0;
      }
      posIncrAttr.setPositionIncrement(posIncr);
      pos += posIncr;
      if (TestPositionIncrement.VERBOSE) {
        System.out.println("term=" + termAttr + " pos=" + pos);
      }
      i++;
      return true;
    } else {
      return false;
    }
  }
}

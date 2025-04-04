package org.apache.lucene.benchmark.byTask.feeds;

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
import java.util.Properties;

import org.apache.lucene.analysis.SimpleAnalyzer;
import org.apache.lucene.analysis.WhitespaceAnalyzer;
import org.apache.lucene.benchmark.BenchmarkTestCase;
import org.apache.lucene.benchmark.byTask.PerfRunData;
import org.apache.lucene.benchmark.byTask.tasks.AddDocTask;
import org.apache.lucene.benchmark.byTask.tasks.CloseIndexTask;
import org.apache.lucene.benchmark.byTask.tasks.CreateIndexTask;
import org.apache.lucene.benchmark.byTask.tasks.TaskSequence;
import org.apache.lucene.benchmark.byTask.utils.Config;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;

/** Tests the functionality of {@link DocMaker}. */
public class DocMakerTest extends BenchmarkTestCase {

  static final class OneDocSource extends ContentSource {

    private boolean finish = false;
    
    @Override
    public void close() throws IOException {
    }

    @Override
    public DocData getNextDocData(DocData docData) throws NoMoreDataException,
        IOException {
      if (finish) {
        throw new NoMoreDataException();
      }
      
      docData.setBody("body");
      docData.setDate("date");
      docData.setTitle("title");
      Properties props = new Properties();
      props.setProperty("key", "value");
      docData.setProps(props);
      finish = true;
      
      return docData;
    }
    
  }

  private void doTestIndexProperties(boolean setIndexProps,
      boolean indexPropsVal, int numExpectedResults) throws Exception {
    Properties props = new Properties();
    
    // Indexing configuration.
    props.setProperty("analyzer", WhitespaceAnalyzer.class.getName());
    props.setProperty("content.source", OneDocSource.class.getName());
    props.setProperty("directory", "RAMDirectory");
    if (setIndexProps) {
      props.setProperty("doc.index.props", Boolean.toString(indexPropsVal));
    }
    
    // Create PerfRunData
    Config config = new Config(props);
    PerfRunData runData = new PerfRunData(config);

    TaskSequence tasks = new TaskSequence(runData, getName(), null, false);
    tasks.addTask(new CreateIndexTask(runData));
    tasks.addTask(new AddDocTask(runData));
    tasks.addTask(new CloseIndexTask(runData));
    tasks.doLogic();
    
    IndexSearcher searcher = new IndexSearcher(runData.getDirectory(), true);
    TopDocs td = searcher.search(new TermQuery(new Term("key", "value")), 10);
    assertEquals(numExpectedResults, td.totalHits);
    searcher.close();
  }
  
  private Document createTestNormsDocument(boolean setNormsProp,
      boolean normsPropVal, boolean setBodyNormsProp, boolean bodyNormsVal)
      throws Exception {
    Properties props = new Properties();
    
    // Indexing configuration.
    props.setProperty("analyzer", WhitespaceAnalyzer.class.getName());
    props.setProperty("content.source", OneDocSource.class.getName());
    props.setProperty("directory", "RAMDirectory");
    if (setNormsProp) {
      props.setProperty("doc.tokenized.norms", Boolean.toString(normsPropVal));
    }
    if (setBodyNormsProp) {
      props.setProperty("doc.body.tokenized.norms", Boolean.toString(bodyNormsVal));
    }
    
    // Create PerfRunData
    Config config = new Config(props);
    
    DocMaker dm = new DocMaker();
    dm.setConfig(config);
    return dm.makeDocument();
  }
  
  /* Tests doc.index.props property. */
  public void testIndexProperties() throws Exception {
    // default is to not index properties.
    doTestIndexProperties(false, false, 0);
    
    // set doc.index.props to false.
    doTestIndexProperties(true, false, 0);
    
    // set doc.index.props to true.
    doTestIndexProperties(true, true, 1);
  }
  
  /* Tests doc.tokenized.norms and doc.body.tokenized.norms properties. */
  public void testNorms() throws Exception {
    
    Document doc;
    
    // Don't set anything, use the defaults
    doc = createTestNormsDocument(false, false, false, false);
    assertTrue(doc.getField(DocMaker.TITLE_FIELD).getOmitNorms());
    assertFalse(doc.getField(DocMaker.BODY_FIELD).getOmitNorms());
    
    // Set norms to false
    doc = createTestNormsDocument(true, false, false, false);
    assertTrue(doc.getField(DocMaker.TITLE_FIELD).getOmitNorms());
    assertFalse(doc.getField(DocMaker.BODY_FIELD).getOmitNorms());
    
    // Set norms to true
    doc = createTestNormsDocument(true, true, false, false);
    assertFalse(doc.getField(DocMaker.TITLE_FIELD).getOmitNorms());
    assertFalse(doc.getField(DocMaker.BODY_FIELD).getOmitNorms());
    
    // Set body norms to false
    doc = createTestNormsDocument(false, false, true, false);
    assertTrue(doc.getField(DocMaker.TITLE_FIELD).getOmitNorms());
    assertTrue(doc.getField(DocMaker.BODY_FIELD).getOmitNorms());
    
    // Set body norms to true
    doc = createTestNormsDocument(false, false, true, true);
    assertTrue(doc.getField(DocMaker.TITLE_FIELD).getOmitNorms());
    assertFalse(doc.getField(DocMaker.BODY_FIELD).getOmitNorms());
  }
  
}

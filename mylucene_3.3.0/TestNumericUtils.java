package org.apache.lucene.util;

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

import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.Random;

public class TestNumericUtils extends LuceneTestCase {

  public void testLongConversionAndOrdering() throws Exception {
    // generate a series of encoded longs, each numerical one bigger than the one before
    String last=null;
    for (long l=-100000L; l<100000L; l++) {
      String act=NumericUtils.longToPrefixCoded(l);
      if (last!=null) {
        // test if smaller
        assertTrue("actual bigger than last", last.compareTo(act) < 0 );
      }
      // test is back and forward conversion works
      assertEquals("forward and back conversion should generate same long", l, NumericUtils.prefixCodedToLong(act));
      // next step
      last=act;
    }
  }

  public void testIntConversionAndOrdering() throws Exception {
    // generate a series of encoded ints, each numerical one bigger than the one before
    String last=null;
    for (int i=-100000; i<100000; i++) {
      String act=NumericUtils.intToPrefixCoded(i);
      if (last!=null) {
        // test if smaller
        assertTrue("actual bigger than last", last.compareTo(act) < 0 );
      }
      // test is back and forward conversion works
      assertEquals("forward and back conversion should generate same int", i, NumericUtils.prefixCodedToInt(act));
      // next step
      last=act;
    }
  }

  public void testLongSpecialValues() throws Exception {
    long[] vals=new long[]{
      Long.MIN_VALUE, Long.MIN_VALUE+1, Long.MIN_VALUE+2, -5003400000000L,
      -4000L, -3000L, -2000L, -1000L, -1L, 0L, 1L, 10L, 300L, 50006789999999999L, Long.MAX_VALUE-2, Long.MAX_VALUE-1, Long.MAX_VALUE
    };
    String[] prefixVals=new String[vals.length];
    
    for (int i=0; i<vals.length; i++) {
      prefixVals[i]=NumericUtils.longToPrefixCoded(vals[i]);
      
      // check forward and back conversion
      assertEquals( "forward and back conversion should generate same long", vals[i], NumericUtils.prefixCodedToLong(prefixVals[i]) );

      // test if decoding values as int fails correctly
      try {
        NumericUtils.prefixCodedToInt(prefixVals[i]);
        fail("decoding a prefix coded long value as int should fail");
      } catch (NumberFormatException e) {
        // worked
      }
    }
    
    // check sort order (prefixVals should be ascending)
    for (int i=1; i<prefixVals.length; i++) {
      assertTrue( "check sort order", prefixVals[i-1].compareTo( prefixVals[i] ) < 0 );
    }
        
    // check the prefix encoding, lower precision should have the difference to original value equal to the lower removed bits
    for (int i=0; i<vals.length; i++) {
      for (int j=0; j<64; j++) {
        long prefixVal=NumericUtils.prefixCodedToLong(NumericUtils.longToPrefixCoded(vals[i], j));
        long mask=(1L << j) - 1L;
        assertEquals( "difference between prefix val and original value for "+vals[i]+" with shift="+j, vals[i] & mask, vals[i]-prefixVal );
      }
    }
  }

  public void testIntSpecialValues() throws Exception {
    int[] vals=new int[]{
      Integer.MIN_VALUE, Integer.MIN_VALUE+1, Integer.MIN_VALUE+2, -64765767,
      -4000, -3000, -2000, -1000, -1, 0, 1, 10, 300, 765878989, Integer.MAX_VALUE-2, Integer.MAX_VALUE-1, Integer.MAX_VALUE
    };
    String[] prefixVals=new String[vals.length];
    
    for (int i=0; i<vals.length; i++) {
      prefixVals[i]=NumericUtils.intToPrefixCoded(vals[i]);
      
      // check forward and back conversion
      assertEquals( "forward and back conversion should generate same int", vals[i], NumericUtils.prefixCodedToInt(prefixVals[i]) );
      
      // test if decoding values as long fails correctly
      try {
        NumericUtils.prefixCodedToLong(prefixVals[i]);
        fail("decoding a prefix coded int value as long should fail");
      } catch (NumberFormatException e) {
        // worked
      }
    }
    
    // check sort order (prefixVals should be ascending)
    for (int i=1; i<prefixVals.length; i++) {
      assertTrue( "check sort order", prefixVals[i-1].compareTo( prefixVals[i] ) < 0 );
    }
    
    // check the prefix encoding, lower precision should have the difference to original value equal to the lower removed bits
    for (int i=0; i<vals.length; i++) {
      for (int j=0; j<32; j++) {
        int prefixVal=NumericUtils.prefixCodedToInt(NumericUtils.intToPrefixCoded(vals[i], j));
        int mask=(1 << j) - 1;
        assertEquals( "difference between prefix val and original value for "+vals[i]+" with shift="+j, vals[i] & mask, vals[i]-prefixVal );
      }
    }
  }

  public void testDoubles() throws Exception {
    double[] vals=new double[]{
      Double.NEGATIVE_INFINITY, -2.3E25, -1.0E15, -1.0, -1.0E-1, -1.0E-2, -0.0, 
      +0.0, 1.0E-2, 1.0E-1, 1.0, 1.0E15, 2.3E25, Double.POSITIVE_INFINITY
    };
    long[] longVals=new long[vals.length];
    
    // check forward and back conversion
    for (int i=0; i<vals.length; i++) {
      longVals[i]=NumericUtils.doubleToSortableLong(vals[i]);
      assertTrue( "forward and back conversion should generate same double", Double.compare(vals[i], NumericUtils.sortableLongToDouble(longVals[i]))==0 );
    }
    
    // check sort order (prefixVals should be ascending)
    for (int i=1; i<longVals.length; i++) {
      assertTrue( "check sort order", longVals[i-1] < longVals[i] );
    }
  }

  public void testFloats() throws Exception {
    float[] vals=new float[]{
      Float.NEGATIVE_INFINITY, -2.3E25f, -1.0E15f, -1.0f, -1.0E-1f, -1.0E-2f, -0.0f, 
      +0.0f, 1.0E-2f, 1.0E-1f, 1.0f, 1.0E15f, 2.3E25f, Float.POSITIVE_INFINITY
    };
    int[] intVals=new int[vals.length];
    
    // check forward and back conversion
    for (int i=0; i<vals.length; i++) {
      intVals[i]=NumericUtils.floatToSortableInt(vals[i]);
      assertTrue( "forward and back conversion should generate same double", Float.compare(vals[i], NumericUtils.sortableIntToFloat(intVals[i]))==0 );
    }
    
    // check sort order (prefixVals should be ascending)
    for (int i=1; i<intVals.length; i++) {
      assertTrue( "check sort order", intVals[i-1] < intVals[i] );
    }
  }
  
  // INFO: Tests for trieCodeLong()/trieCodeInt() not needed because implicitely tested by range filter tests
  
  /** Note: The neededBounds Iterable must be unsigned (easier understanding what's happening) */
  private void assertLongRangeSplit(final long lower, final long upper, int precisionStep,
    final boolean useBitSet, final Iterable<Long> expectedBounds, final Iterable<Integer> expectedShifts
  ) throws Exception {
    final OpenBitSet bits=useBitSet ? new OpenBitSet(upper-lower+1) : null;
    final Iterator<Long> neededBounds = (expectedBounds == null) ? null : expectedBounds.iterator();
    final Iterator<Integer> neededShifts = (expectedShifts == null) ? null : expectedShifts.iterator();

    NumericUtils.splitLongRange(new NumericUtils.LongRangeBuilder() {
      @Override
      public void addRange(long min, long max, int shift) {
        assertTrue("min, max should be inside bounds", min>=lower && min<=upper && max>=lower && max<=upper);
        if (useBitSet) for (long l=min; l<=max; l++) {
          assertFalse("ranges should not overlap", bits.getAndSet(l-lower) );
          // extra exit condition to prevent overflow on MAX_VALUE
          if (l == max) break;
        }
        if (neededBounds == null || neededShifts == null)
          return;
        // make unsigned longs for easier display and understanding
        min ^= 0x8000000000000000L;
        max ^= 0x8000000000000000L;
        //System.out.println("0x"+Long.toHexString(min>>>shift)+"L,0x"+Long.toHexString(max>>>shift)+"L)/*shift="+shift+"*/,");
        assertEquals( "shift", neededShifts.next().intValue(), shift);
        assertEquals( "inner min bound", neededBounds.next().longValue(), min>>>shift);
        assertEquals( "inner max bound", neededBounds.next().longValue(), max>>>shift);
      }
    }, precisionStep, lower, upper);
    
    if (useBitSet) {
      // after flipping all bits in the range, the cardinality should be zero
      bits.flip(0,upper-lower+1);
      assertTrue("The sub-range concenated should match the whole range", bits.isEmpty());
    }
  }
  
  /** LUCENE-2541: NumericRangeQuery errors with endpoints near long min and max values */
  public void testLongExtremeValues() throws Exception {
    // upper end extremes
    assertLongRangeSplit(Long.MAX_VALUE, Long.MAX_VALUE, 1, true, Arrays.asList(
      0xffffffffffffffffL,0xffffffffffffffffL
    ), Arrays.asList(
      0
    ));
    assertLongRangeSplit(Long.MAX_VALUE, Long.MAX_VALUE, 2, true, Arrays.asList(
      0xffffffffffffffffL,0xffffffffffffffffL
    ), Arrays.asList(
      0
    ));
    assertLongRangeSplit(Long.MAX_VALUE, Long.MAX_VALUE, 4, true, Arrays.asList(
      0xffffffffffffffffL,0xffffffffffffffffL
    ), Arrays.asList(
      0
    ));
    assertLongRangeSplit(Long.MAX_VALUE, Long.MAX_VALUE, 6, true, Arrays.asList(
      0xffffffffffffffffL,0xffffffffffffffffL
    ), Arrays.asList(
      0
    ));
    assertLongRangeSplit(Long.MAX_VALUE, Long.MAX_VALUE, 8, true, Arrays.asList(
      0xffffffffffffffffL,0xffffffffffffffffL
    ), Arrays.asList(
      0
    ));
    assertLongRangeSplit(Long.MAX_VALUE, Long.MAX_VALUE, 64, true, Arrays.asList(
      0xffffffffffffffffL,0xffffffffffffffffL
    ), Arrays.asList(
      0
    ));

    assertLongRangeSplit(Long.MAX_VALUE-0xfL, Long.MAX_VALUE, 4, true, Arrays.asList(
      0xfffffffffffffffL,0xfffffffffffffffL
    ), Arrays.asList(
      4
    ));
    assertLongRangeSplit(Long.MAX_VALUE-0x10L, Long.MAX_VALUE, 4, true, Arrays.asList(
      0xffffffffffffffefL,0xffffffffffffffefL,
      0xfffffffffffffffL,0xfffffffffffffffL
    ), Arrays.asList(
      0, 4
    ));

    // lower end extremes
    assertLongRangeSplit(Long.MIN_VALUE, Long.MIN_VALUE, 1, true, Arrays.asList(
      0x0000000000000000L,0x0000000000000000L
    ), Arrays.asList(
      0
    ));
    assertLongRangeSplit(Long.MIN_VALUE, Long.MIN_VALUE, 2, true, Arrays.asList(
      0x0000000000000000L,0x0000000000000000L
    ), Arrays.asList(
      0
    ));
    assertLongRangeSplit(Long.MIN_VALUE, Long.MIN_VALUE, 4, true, Arrays.asList(
      0x0000000000000000L,0x0000000000000000L
    ), Arrays.asList(
      0
    ));
    assertLongRangeSplit(Long.MIN_VALUE, Long.MIN_VALUE, 6, true, Arrays.asList(
      0x0000000000000000L,0x0000000000000000L
    ), Arrays.asList(
      0
    ));
    assertLongRangeSplit(Long.MIN_VALUE, Long.MIN_VALUE, 8, true, Arrays.asList(
      0x0000000000000000L,0x0000000000000000L
    ), Arrays.asList(
      0
    ));
    assertLongRangeSplit(Long.MIN_VALUE, Long.MIN_VALUE, 64, true, Arrays.asList(
      0x0000000000000000L,0x0000000000000000L
    ), Arrays.asList(
      0
    ));

    assertLongRangeSplit(Long.MIN_VALUE, Long.MIN_VALUE+0xfL, 4, true, Arrays.asList(
      0x000000000000000L,0x000000000000000L
    ), Arrays.asList(
      4
    ));
    assertLongRangeSplit(Long.MIN_VALUE, Long.MIN_VALUE+0x10L, 4, true, Arrays.asList(
      0x0000000000000010L,0x0000000000000010L,
      0x000000000000000L,0x000000000000000L
    ), Arrays.asList(
      0, 4
    ));
  }
  
  public void testRandomSplit() throws Exception {
    long num = (long) atLeast(10);
    for (long i=0; i < num; i++) {
      executeOneRandomSplit(random);
    }
  }
  
  private void executeOneRandomSplit(final Random random) throws Exception {
    long lower = randomLong(random);
    long len = random.nextInt(16384*1024); // not too large bitsets, else OOME!
    while (lower + len < lower) { // overflow
      lower >>= 1;
    }
    assertLongRangeSplit(lower, lower + len, random.nextInt(64) + 1, true, null, null);
  }
  
  private long randomLong(final Random random) {
    long val;
    switch(random.nextInt(4)) {
      case 0:
        val = 1L << (random.nextInt(63)); //  patterns like 0x000000100000 (-1 yields patterns like 0x0000fff)
        break;
      case 1:
        val = -1L << (random.nextInt(63)); // patterns like 0xfffff00000
        break;
      default:
        val = random.nextLong();
    }

    val += random.nextInt(5)-2;

    if (random.nextBoolean()) {
      if (random.nextBoolean()) val += random.nextInt(100)-50;
      if (random.nextBoolean()) val = ~val;
      if (random.nextBoolean()) val = val<<1;
      if (random.nextBoolean()) val = val>>>1;
    }

    return val;
  }
  
  public void testSplitLongRange() throws Exception {
    // a hard-coded "standard" range
    assertLongRangeSplit(-5000L, 9500L, 4, true, Arrays.asList(
      0x7fffffffffffec78L,0x7fffffffffffec7fL,
      0x8000000000002510L,0x800000000000251cL,
      0x7fffffffffffec8L, 0x7fffffffffffecfL,
      0x800000000000250L, 0x800000000000250L,
      0x7fffffffffffedL,  0x7fffffffffffefL,
      0x80000000000020L,  0x80000000000024L,
      0x7ffffffffffffL,   0x8000000000001L
    ), Arrays.asList(
      0, 0,
      4, 4,
      8, 8,
      12
    ));
    
    // the same with no range splitting
    assertLongRangeSplit(-5000L, 9500L, 64, true, Arrays.asList(
      0x7fffffffffffec78L,0x800000000000251cL
    ), Arrays.asList(
      0
    ));
    
    // this tests optimized range splitting, if one of the inner bounds
    // is also the bound of the next lower precision, it should be used completely
    assertLongRangeSplit(0L, 1024L+63L, 4, true, Arrays.asList(
      0x800000000000040L, 0x800000000000043L,
      0x80000000000000L,  0x80000000000003L
    ), Arrays.asList(
      4, 8
    ));
    
    // the full long range should only consist of a lowest precision range; no bitset testing here, as too much memory needed :-)
    assertLongRangeSplit(Long.MIN_VALUE, Long.MAX_VALUE, 8, false, Arrays.asList(
      0x00L,0xffL
    ), Arrays.asList(
      56
    ));

    // the same with precisionStep=4
    assertLongRangeSplit(Long.MIN_VALUE, Long.MAX_VALUE, 4, false, Arrays.asList(
      0x0L,0xfL
    ), Arrays.asList(
      60
    ));

    // the same with precisionStep=2
    assertLongRangeSplit(Long.MIN_VALUE, Long.MAX_VALUE, 2, false, Arrays.asList(
      0x0L,0x3L
    ), Arrays.asList(
      62
    ));

    // the same with precisionStep=1
    assertLongRangeSplit(Long.MIN_VALUE, Long.MAX_VALUE, 1, false, Arrays.asList(
      0x0L,0x1L
    ), Arrays.asList(
      63
    ));

    // a inverse range should produce no sub-ranges
    assertLongRangeSplit(9500L, -5000L, 4, false, Collections.<Long>emptyList(), Collections.<Integer>emptyList());    

    // a 0-length range should reproduce the range itsself
    assertLongRangeSplit(9500L, 9500L, 4, false, Arrays.asList(
      0x800000000000251cL,0x800000000000251cL
    ), Arrays.asList(
      0
    ));
  }

  /** Note: The neededBounds Iterable must be unsigned (easier understanding what's happening) */
  private void assertIntRangeSplit(final int lower, final int upper, int precisionStep,
    final boolean useBitSet, final Iterable<Integer> expectedBounds, final Iterable<Integer> expectedShifts
  ) throws Exception {
    final OpenBitSet bits=useBitSet ? new OpenBitSet(upper-lower+1) : null;
    final Iterator<Integer> neededBounds = (expectedBounds == null) ? null : expectedBounds.iterator();
    final Iterator<Integer> neededShifts = (expectedShifts == null) ? null : expectedShifts.iterator();
    
    NumericUtils.splitIntRange(new NumericUtils.IntRangeBuilder() {
      @Override
      public void addRange(int min, int max, int shift) {
        assertTrue("min, max should be inside bounds", min>=lower && min<=upper && max>=lower && max<=upper);
        if (useBitSet) for (int i=min; i<=max; i++) {
          assertFalse("ranges should not overlap", bits.getAndSet(i-lower) );
          // extra exit condition to prevent overflow on MAX_VALUE
          if (i == max) break;
        }
        if (neededBounds == null)
          return;
        // make unsigned ints for easier display and understanding
        min ^= 0x80000000;
        max ^= 0x80000000;
        //System.out.println("0x"+Integer.toHexString(min>>>shift)+",0x"+Integer.toHexString(max>>>shift)+")/*shift="+shift+"*/,");
        assertEquals( "shift", neededShifts.next().intValue(), shift);
        assertEquals( "inner min bound", neededBounds.next().intValue(), min>>>shift);
        assertEquals( "inner max bound", neededBounds.next().intValue(), max>>>shift);
      }
    }, precisionStep, lower, upper);
    
    if (useBitSet) {
      // after flipping all bits in the range, the cardinality should be zero
      bits.flip(0,upper-lower+1);
      assertTrue("The sub-range concenated should match the whole range", bits.isEmpty());
    }
  }
  
  public void testSplitIntRange() throws Exception {
    // a hard-coded "standard" range
    assertIntRangeSplit(-5000, 9500, 4, true, Arrays.asList(
      0x7fffec78,0x7fffec7f,
      0x80002510,0x8000251c,
      0x7fffec8, 0x7fffecf,
      0x8000250, 0x8000250,
      0x7fffed,  0x7fffef,
      0x800020,  0x800024,
      0x7ffff,   0x80001
    ), Arrays.asList(
      0, 0,
      4, 4,
      8, 8,
      12
    ));
    
    // the same with no range splitting
    assertIntRangeSplit(-5000, 9500, 32, true, Arrays.asList(
      0x7fffec78,0x8000251c
    ), Arrays.asList(
      0
    ));
    
    // this tests optimized range splitting, if one of the inner bounds
    // is also the bound of the next lower precision, it should be used completely
    assertIntRangeSplit(0, 1024+63, 4, true, Arrays.asList(
      0x8000040, 0x8000043,
      0x800000,  0x800003
    ), Arrays.asList(
      4, 8
    ));
    
    // the full int range should only consist of a lowest precision range; no bitset testing here, as too much memory needed :-)
    assertIntRangeSplit(Integer.MIN_VALUE, Integer.MAX_VALUE, 8, false, Arrays.asList(
      0x00,0xff
    ), Arrays.asList(
      24
    ));

    // the same with precisionStep=4
    assertIntRangeSplit(Integer.MIN_VALUE, Integer.MAX_VALUE, 4, false, Arrays.asList(
      0x0,0xf
    ), Arrays.asList(
      28
    ));

    // the same with precisionStep=2
    assertIntRangeSplit(Integer.MIN_VALUE, Integer.MAX_VALUE, 2, false, Arrays.asList(
      0x0,0x3
    ), Arrays.asList(
      30
    ));

    // the same with precisionStep=1
    assertIntRangeSplit(Integer.MIN_VALUE, Integer.MAX_VALUE, 1, false, Arrays.asList(
      0x0,0x1
    ), Arrays.asList(
      31
    ));

    // a inverse range should produce no sub-ranges
    assertIntRangeSplit(9500, -5000, 4, false, Collections.<Integer>emptyList(), Collections.<Integer>emptyList());    

    // a 0-length range should reproduce the range itsself
    assertIntRangeSplit(9500, 9500, 4, false, Arrays.asList(
      0x8000251c,0x8000251c
    ), Arrays.asList(
      0
    ));
  }

}

/* ------------------------------------------------------------------
 * This source code, its documentation and all appendant files
 * are protected by copyright law. All rights reserved.
 *
 * Copyright, 2003 - 2008
 * University of Konstanz, Germany
 * and KNIME GmbH, Konstanz, Germany
 *
 * You may not modify, publish, transmit, transfer or sell, reproduce,
 * create derivative works from, distribute, perform, display, or in
 * any way exploit any of the content, in whole or in part, except as
 * otherwise expressly permitted in writing by the copyright owner or
 * as specified in the license file distributed with this product.
 *
 * If you have any questions please contact the copyright holder:
 * website: www.knime.org
 * email: contact@knime.org
 * ---------------------------------------------------------------------
 *
 * History
 *   27.08.2008 (ohl): created
 */
package org.knime.core.data.collection.bitvector;

import java.util.Arrays;

/**
 * Stores Zeros and Ones in a vector, i.e. with fixed positions. The vector has
 * a fixed length. <br />
 * Implementation assumes that the vector is only sparsely populated with '1's.
 * It stores the indices of the ones. For densely populated vectors
 * {@link DenseBitVector} is more suitable.<br />
 * The length of the vector is restricted to {@link Long#MAX_VALUE} (i.e.
 * 9223372036854775807). The number of ones that can be stored is limited to
 * {@link Integer#MAX_VALUE} (which is 2147483647), in which case it uses about
 * 16Gbyte of memory.
 *
 * @author ohl, University of Konstanz
 */
public class SparseBitVector {

    // make sure the length of the array is always a power of 2.
    private long[] m_idxStorage;

    // the index of the last used position in the array. -1 if no bit is set.
    private int m_lastIdx;

    private final long m_length;

    /**
     * Creates a new vector with (initially) space for 64 ones and of the
     * specified length.
     *
     * @param length the length of the vector to create
     */
    public SparseBitVector(final long length) {
        this(length, 64);
    }

    /**
     * Creates a new vector of the specified length and with (initially) space
     * for the specified number of ones.
     *
     * @param length the length of the vector to create
     * @param initialCapacity space will be allocated to store that many ones
     */
    public SparseBitVector(final long length, final int initialCapacity) {
        if (length < 0) {
            throw new IllegalArgumentException(
                    "Length of a BitVector can't be negative.");
        }
        m_length = length;

        // be sure to start with a power of two (and not zero!)
        int c = 1;
        while (c < initialCapacity && c < m_length) {
            c <<= 1;
        }
        m_idxStorage = new long[c];
        m_lastIdx = -1;
    }

    /**
     * Creates a new instance by taking over the initialization from the passed
     * array. The numbers in the array are considered indices of the bits set to
     * one in the vector. The array must be sorted! The lowest bit index must be
     * stored at array index zero. The array must be build like the one returned
     * by the {@link #getAllOneIndices()} method.
     *
     * @param oneIndices the array containing the indices of the ones. MUST be
     *            sorted (lowest index first).
     * @param length the length of the vector. Indices must be smaller than this
     *            number.
     * @throws IllegalArgumentException if length is negative or if the array
     *             contains negative numbers or numbers larger than length - or
     *             if the array is not sorted!
     *
     */
    public SparseBitVector(final long length, final long[] oneIndices) {
        this(length, oneIndices.length);

        if (length < oneIndices.length) {
            throw new IllegalArgumentException("Can't init a vector of length "
                    + length + " with " + oneIndices.length + " ones");
        }
        int idx = -1;
        long lastVal = -1;
        int maxArrayIdx = oneIndices.length - 1;

        while (idx < maxArrayIdx) {
            idx++;
            if (oneIndices[idx] >= length) {
                throw new IllegalArgumentException("Initialization array"
                        + " contains index out range at array index " + idx
                        + " (vector length=" + length + ", index="
                        + oneIndices[idx] + ")");
            }
            if (oneIndices[idx] < 0) {
                throw new IllegalArgumentException("Initialization array"
                        + " contains a negative index at array index " + idx
                        + "(index=" + oneIndices[idx] + ")");
            }
            if (oneIndices[idx] <= lastVal) {
                throw new IllegalArgumentException("Initialization array"
                        + " is not sorted at array index " + idx
                        + " (previousVal=" + lastVal + ", indexVal="
                        + oneIndices[idx] + ")");
            }

            m_idxStorage[idx] = oneIndices[idx];
            lastVal = oneIndices[idx];
        }

        m_lastIdx = idx;

    }

    /**
     * Returns the number of bits stored in this vector.
     *
     * @return the length of the vector.
     */
    public long length() {
        return m_length;
    }

    /**
     * Frees unused memory in the vector. If a vector loses a lot of ones the
     * used storage could be reduced (as only the indices of ones are stores).
     */
    public void shrink() {
        int newLength = m_idxStorage.length;
        if (m_lastIdx <= 0) {
            // never create a zero-length array (wouldn't grow anymore!)
            newLength = 1;
        } else {
            assert Long.bitCount(newLength) == 1;
            while (newLength > m_lastIdx) {
                newLength >>>= 1;
            }
            newLength <<= 1;
        }
        if (newLength < m_idxStorage.length) {
            m_idxStorage = Arrays.copyOf(m_idxStorage, newLength);
        }
    }

    /**
     * Sets the bit at the specified index to the new value.
     *
     * @param bitIdx the index of the bit to set or clear
     * @param value if true, the specified bit will be set, otherwise it will be
     *            cleared.
     * @throws ArrayIndexOutOfBoundsException if the index is negative or larger
     *             than the size of the vector
     */
    public void set(final long bitIdx, final boolean value) {
        if (value) {
            set(bitIdx);
        } else {
            clear(bitIdx);
        }
    }

    /**
     * Sets the bit at the specified index to zero.
     *
     * @param bitIdx the index of the bit to clear.
     * @throws ArrayIndexOutOfBoundsException if the index is negative or larger
     *             than the size of the vector
     */
    public void set(final long bitIdx) {
        assert (checkConsistency() == null);
        if (bitIdx >= m_length) {
            throw new ArrayIndexOutOfBoundsException("Index ('" + bitIdx
                    + "') too large for vector of length " + m_length);
        }
        if (bitIdx < 0) {
            throw new ArrayIndexOutOfBoundsException("Index of the bit to set"
                    + " can't be negative");
        }
        int storageIdx;
        if (m_lastIdx == -1 || m_idxStorage[0] > bitIdx) {
            // no bit set yet - or only higher bits set
            storageIdx = 0;
        } else if (bitIdx > m_idxStorage[m_lastIdx]) {
            // might be faster when they set the bits in order
            storageIdx = m_lastIdx + 1;
        } else {
            storageIdx =
                    Arrays.binarySearch(m_idxStorage, 0, m_lastIdx + 1, bitIdx);
            if (storageIdx >= 0) {
                // already set
                return;
            }
            // the bit is not set yet - insert it
            storageIdx = -(storageIdx + 1);
        }

        m_lastIdx++;
        long[] newStorage = m_idxStorage;

        // ensure capacity
        if (m_idxStorage.length <= m_lastIdx) {
            if (m_idxStorage.length == Integer.MAX_VALUE) {
                throw new IllegalArgumentException(
                        "The capacity of the sparce bit vector is exceeded."
                                + " Too many bits are set.");
            }
            assert m_idxStorage.length > 0;
            newStorage = new long[m_idxStorage.length << 1];
            // copy the content up to the new index
            System.arraycopy(m_idxStorage, 0, newStorage, 0, storageIdx);
        }

        // shift the old content
        System.arraycopy(m_idxStorage, storageIdx, newStorage, storageIdx + 1,
                m_lastIdx - storageIdx);
        // insert the new bit index
        newStorage[storageIdx] = bitIdx;
        // in case we re-allocated
        m_idxStorage = newStorage;

        assert (checkConsistency() == null);
    }

    /**
     * Sets the bit at the specified index to one.
     *
     * @param bitIdx the index of the bit to set.
     * @throws ArrayIndexOutOfBoundsException if the index is negative or larger
     *             than the size of the vector
     */
    public void clear(final long bitIdx) {
        assert (checkConsistency() == null);
        if (bitIdx >= m_length) {
            throw new ArrayIndexOutOfBoundsException("Index ('" + bitIdx
                    + "') too large for vector of length " + m_length);
        }
        if (bitIdx < 0) {
            throw new ArrayIndexOutOfBoundsException("Index of the bit to set"
                    + " can't be negative");
        }
        if (m_lastIdx == -1 || m_idxStorage[m_lastIdx] < bitIdx
                || m_idxStorage[0] > bitIdx) {
            // no bit set yet - or only lower or higher bits set
            return;
        }

        int storageIdx =
                Arrays.binarySearch(m_idxStorage, 0, m_lastIdx + 1, bitIdx);
        if (storageIdx < 0) {
            // bit is not set
            return;
        }

        if (storageIdx < m_lastIdx) {
            // shift the higher content down over the bit to clear
            System.arraycopy(m_idxStorage, storageIdx + 1, m_idxStorage,
                    storageIdx, m_lastIdx - storageIdx);
        }

        m_lastIdx--;
        assert (checkConsistency() == null);
    }

    /**
     * Number of bits set in this bit vector.
     *
     * @return the number of ones in this vector
     */
    public int cardinality() {
        return m_lastIdx + 1;
    }

    /**
     * Returns true if no bits are set in this bit vector.
     *
     * @return true if no bits are set in this bit vector.
     */
    public boolean isEmpty() {
        return m_lastIdx == -1;
    }

    /**
     * Returns true, if this and the argument vector have at least one bit set
     * at the same position.
     *
     * @param bv the vector to test
     * @return true, if this and the argument vector have at least one bit set
     *         at the same position.
     */
    public boolean intersects(final SparseBitVector bv) {
        if (m_lastIdx < 0 || bv.m_lastIdx < 0) {
            // empty vector doesn't intersect
            return false;
        }
        if (m_idxStorage[0] > bv.m_idxStorage[bv.m_lastIdx]
                || m_idxStorage[m_lastIdx] < bv.m_idxStorage[0]) {
            // ranges of bits set don't intersect
            return false;
        }
        int thisIdx = 0;
        int bvIdx = 0;
        while (thisIdx <= m_lastIdx && bvIdx <= bv.m_lastIdx) {
            if (m_idxStorage[thisIdx] == bv.m_idxStorage[bvIdx]) {
                return true;
            }
            if (m_idxStorage[thisIdx] < bv.m_idxStorage[bvIdx]) {
                thisIdx++;
            } else {
                bvIdx++;
            }
        }
        return false;
    }

    /**
     * Returns true if the bit at the specified index is set. False otherwise.
     *
     * @param bitIdx the index of the bit to test.
     * @return <code>true</code> if the specified bit is set,
     *         <code>false</code> otherwise
     * @throws ArrayIndexOutOfBoundsException if the index is larger than the
     *             length of the vector
     */
    public boolean get(final long bitIdx) {
        assert (checkConsistency() == null);
        if (bitIdx >= m_length) {
            throw new ArrayIndexOutOfBoundsException("Index ('" + bitIdx
                    + "') too large for vector of length " + m_length);
        }
        if (bitIdx < 0) {
            throw new ArrayIndexOutOfBoundsException("Index of the bit to get "
                    + " can't be negative");
        }
        if (m_lastIdx == -1 || m_idxStorage[0] > bitIdx
                || m_idxStorage[m_lastIdx] < bitIdx) {
            // no bit set yet - or only higher or lower bits set
            return false;
        }
        return Arrays.binarySearch(m_idxStorage, 0, m_lastIdx + 1, bitIdx) >= 0;
    }

    /**
     * Finds the next bit set to one on or after the specified index. Returns an
     * index larger than or equal the provided index, or -1 if no bit is set
     * after the startIdx. (This is the only method (and the #nextClearBit)
     * where it is okay to pass an index larger than the length of the vector.)
     *
     * @param startIdx the first index to look for '1's. (It is allowed to pass
     *            an index larger then the vector's length.)
     * @return the index of the next bit set to one, which is on or after the
     *         provided startIdx.
     * @throws ArrayIndexOutOfBoundsException if the specified startIdx is
     *             negative
     */
    public long nextSetBit(final int startIdx) {
        if (startIdx < 0) {
            throw new ArrayIndexOutOfBoundsException("Starting index"
                    + " can't be negative");
        }
        if (m_lastIdx < 0 || m_idxStorage[m_lastIdx] < startIdx) {
            // no bit set - or none above or on startIdx
            return -1;
        }
        if (startIdx <= m_idxStorage[0]) {
            return m_idxStorage[0];
        }

        int startAddr =
                Arrays.binarySearch(m_idxStorage, 0, m_lastIdx + 1, startIdx);
        if (startAddr < 0) {
            startAddr = -(startAddr + 1);
            if (startAddr == m_lastIdx + 1) {
                // there are only smaller bit addresses in the storage
                return -1;
            }
            // otherwise it points to the "insert point"
            return m_idxStorage[startAddr];
        } else {
            // found the startIdx.
            return startIdx;
        }
    }

    /**
     * Finds the next bit not set (that is '0') on or after the specified index.
     * Returns an index larger than or equal the provided index, or -1 if no bit
     * is cleared after the startIdx. (This is the only method (and the
     * #nextSetBit) where it is okay to pass an index larger than the length of
     * the vector.)
     *
     * @param startIdx the first index to look for '0's.
     * @return the index of the next cleared bit, which is on or after the
     *         provided startIdx. Or -1 if the vector contains no zero anymore.
     * @throws ArrayIndexOutOfBoundsException if the specified startIdx negative
     */
    public long nextClearBit(final int startIdx) {
        if (startIdx < 0) {
            throw new ArrayIndexOutOfBoundsException("Starting index"
                    + " can't be negative");
        }
        if (m_lastIdx < 0 || startIdx >= m_length) {
            return -1;
        }
        if (m_idxStorage[m_lastIdx] < startIdx || startIdx < m_idxStorage[0]) {
            // startIdx outside the range
            return startIdx;
        }

        int startAddr =
                Arrays.binarySearch(m_idxStorage, 0, m_lastIdx + 1, startIdx);

        if (startAddr < 0) {
            // startIdx not contained - return it as zero bit
            return startIdx;
        }

        long idx = m_idxStorage[startAddr];
        // now find the next index not in the storage
        while (startAddr < m_lastIdx) {
            idx++;
            startAddr++;
            if (idx != m_idxStorage[startAddr]) {
                return idx;
            }
        }
        // all numbers from startAddr are continuously stored (i.e. set to one)
        assert (idx == m_idxStorage[m_lastIdx]);
        return idx++;
    }

    /**
     * Creates and returns a new bit vector whose bits are set at positions
     * where both, this and the argument vector have their bits set. The length
     * of the new vector is the maximum of the length of this and the argument.
     *
     * @param bv the vector to AND this one with
     * @return a new instance containing the result of the AND operation
     */
    public SparseBitVector and(final SparseBitVector bv) {

        SparseBitVector result =
                new SparseBitVector(Math.max(m_length, bv.m_length), Math.min(
                        cardinality(), bv.cardinality()));

        int thisIdx = 0;
        int bvIdx = 0;
        int resultIdx = -1;
        while (thisIdx <= m_lastIdx && bvIdx <= bv.m_lastIdx) {
            if (m_idxStorage[thisIdx] == bv.m_idxStorage[bvIdx]) {
                // index is set in both arguments
                result.m_idxStorage[++resultIdx] = m_idxStorage[thisIdx];
            }
            if (m_idxStorage[thisIdx] < bv.m_idxStorage[bvIdx]) {
                thisIdx++;
            } else {
                bvIdx++;
            }
        }
        result.m_lastIdx = resultIdx;
        return result;
    }

    /**
     * Creates and returns a new bit vector whose bits are set at positions
     * where at least one of the vectors (this or the argument vector) have a
     * bit set. The length of the new vector is the maximum of the length of
     * this and the argument.
     *
     * @param bv the vector to OR this one with
     * @return a new instance containing the result of the OR operation
     */
    public SparseBitVector or(final SparseBitVector bv) {

        SparseBitVector result =
                new SparseBitVector(Math.max(m_length, bv.m_length), Math.max(
                        cardinality(), bv.cardinality()));

        int thisIdx = 0;
        int bvIdx = 0;
        int resultIdx = -1;
        while (thisIdx <= m_lastIdx && bvIdx <= bv.m_lastIdx) {
            if (m_idxStorage[thisIdx] == bv.m_idxStorage[bvIdx]) {
                // index is set in both arguments
                result.m_idxStorage[++resultIdx] = m_idxStorage[thisIdx];
                thisIdx++;
                bvIdx++;
            } else if (m_idxStorage[thisIdx] < bv.m_idxStorage[bvIdx]) {
                result.m_idxStorage[++resultIdx] = m_idxStorage[thisIdx];
                thisIdx++;
            } else {
                result.m_idxStorage[++resultIdx] = bv.m_idxStorage[bvIdx];
                bvIdx++;
            }
        }
        // copy the longer one into the result
        if (thisIdx < m_lastIdx) {
            System.arraycopy(m_idxStorage, thisIdx, result.m_idxStorage,
                    resultIdx + 1, m_lastIdx - thisIdx + 1);
            resultIdx += m_lastIdx - thisIdx + 1;
        }
        if (bvIdx < bv.m_lastIdx) {
            System.arraycopy(bv.m_idxStorage, bvIdx, result.m_idxStorage,
                    resultIdx + 1, bv.m_lastIdx - bvIdx + 1);
            resultIdx += bv.m_lastIdx - bvIdx + 1;
        }
        result.m_lastIdx = resultIdx;
        return result;
    }

    /**
     * Creates and returns a new bit vector whose bits are set at positions
     * where (exactly) one of the vectors (this or the argument vector) have a
     * bit set. The length of the new vector is the maximum of the length of
     * this and the argument.
     *
     * @param bv the vector to XOR this one with
     * @return a new instance containing the result of the XOR operation
     */
    public SparseBitVector xor(final SparseBitVector bv) {

        SparseBitVector result =
                new SparseBitVector(Math.max(m_length, bv.m_length), Math.max(
                        cardinality(), bv.cardinality()));

        int thisIdx = 0;
        int bvIdx = 0;
        int resultIdx = -1;
        while (thisIdx <= m_lastIdx && bvIdx <= bv.m_lastIdx) {
            if (m_idxStorage[thisIdx] == bv.m_idxStorage[bvIdx]) {
                // index set in both arguments - don't store it in the result
                thisIdx++;
                bvIdx++;
            } else if (m_idxStorage[thisIdx] < bv.m_idxStorage[bvIdx]) {
                result.m_idxStorage[++resultIdx] = m_idxStorage[thisIdx];
                thisIdx++;
            } else {
                result.m_idxStorage[++resultIdx] = bv.m_idxStorage[bvIdx];
                bvIdx++;
            }
        }
        // copy the longer one into the result
        if (thisIdx < m_lastIdx) {
            System.arraycopy(m_idxStorage, thisIdx, result.m_idxStorage,
                    resultIdx + 1, m_lastIdx - thisIdx + 1);
            resultIdx += m_lastIdx - thisIdx + 1;
        }
        if (bvIdx < bv.m_lastIdx) {
            System.arraycopy(bv.m_idxStorage, bvIdx, result.m_idxStorage,
                    resultIdx + 1, bv.m_lastIdx - bvIdx + 1);
            resultIdx += bv.m_lastIdx - bvIdx + 1;
        }

        result.m_lastIdx = resultIdx;
        return result;
    }

    /**
     * Creates and returns a new bit vector that contains copies of both (this
     * and the argument vector). The argument vector is appended at the end of
     * this vector, i.e. it's bit with index zero will be stored at index
     * "length-of-this-vector" in the result vector. The length of the result is
     * the length of this plus the length of the argument vector.
     *
     * @param bv the vector to append at the end of this
     * @return a new instance containing both vectors concatenated
     */
    public SparseBitVector concatenate(final SparseBitVector bv) {
        SparseBitVector result =
                new SparseBitVector(m_length + bv.m_length, cardinality()
                        + bv.cardinality());
        System
                .arraycopy(m_idxStorage, 0, result.m_idxStorage, 0,
                        m_lastIdx + 1);
        // copy bv's indices and add the length of this to each
        int resIdx = m_lastIdx + 1;
        for (int bvIdx = 0; bvIdx <= bv.m_lastIdx; bvIdx++) {
            result.m_idxStorage[resIdx] = bv.m_idxStorage[bvIdx] + m_length;
            resIdx++;
        }

        assert resIdx == m_lastIdx + bv.m_lastIdx + 1 + 1;

        result.m_lastIdx = m_lastIdx + bv.m_lastIdx + 1;
        return result;

    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int hashCode() {
        int hash = (int)(m_length ^ (m_length >>> 32));
        for (int i = 0; i <= m_lastIdx; i++) {
            long idx = m_idxStorage[i];
            hash = 31 * hash + ((int)(idx ^ (idx >>> 32)));
        }
        return hash;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean equals(final Object obj) {
        if (obj instanceof SparseBitVector) {
            SparseBitVector s = (SparseBitVector)obj;
            if (s.m_length != m_length) {
                return false;
            }
            if (s.m_lastIdx != m_lastIdx) {
                return false;
            }
            if (m_lastIdx == -1) {
                return true;
            }
            for (int i = 0; i <= m_lastIdx; i++) {
                if (m_idxStorage[i] != s.m_idxStorage[i]) {
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    /**
     * Returns a string containing (comma separated) indices of the bits set in
     * this vector. The number of bit indices added to the string is limited to
     * 30000. If the output is truncated, the string ends on &quot;... }&quot;
     *
     * @return a string containing (comma separated) indices of the bits set in
     *         this vector.
     */
    @Override
    public String toString() {
        int max = Math.min(30000, m_lastIdx + 1);
        StringBuilder result = new StringBuilder(max * 5);
        result.append("{");
        for (int i = 0; i < max; i++) {
            result.append(m_idxStorage[i]).append(", ");
        }
        if (max < m_lastIdx) {
            result.append("... ");
        } else if (result.length() > 2) {
            result.delete(result.length() - 2, result.length());
        }
        result.append('}');
        return result.toString();
    }

    /**
     * There are certain conditions the implementation depends on. They are all
     * checked in here. Normally the method should return null. If it doesn't
     * something is fishy and a error message is returned. NOTE: This method is
     * not cheap! It should be called in an assert statement only.
     *
     * @return the error message, or null if everything is alright
     */
    private String checkConsistency() {
        if (m_length < 0) {
            return "vector's length is negative!";
        }
        if (m_lastIdx >= m_length) {
            return "m_lastIdx not less than m_length!";
        }

        // make sure array is sorted
        long lastVal = -1;
        for (int i = 0; i <= m_lastIdx; i++) {
            if (m_idxStorage[i] <= lastVal) {
                return "Index at position " + i + "(" + m_idxStorage[i]
                        + ") is not larger then the one at " + (i - 1) + "("
                        + lastVal + ")";
            }
            lastVal = m_idxStorage[i];
        }

        return null;
    }

    /**
     * Returns a copy of the internal storage of all bit indices. The array
     * contains the sorted indices of all '1's in the vector. The length of the
     * returned array is the cardinality of the vector.
     *
     * @return a copy of the internal representation of the bits in this vector.
     */
    public long[] getAllOneIndices() {
        return Arrays.copyOf(m_idxStorage, m_lastIdx + 1);
    }

}


/*
 * Copyright (C) SAIC VOLKSWAGEN Automotive Co., Ltd(SVW). All Rights Reserved.
 * This software is the confidential and proprietary information of SVW.
 * You shall not disclose such Confidential Information and shall use it only in
 * accordance with the terms of the license agreement you entered into with SVW.
 * Some implementations are refer to open source projects with permissive
 * open source licenses.
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PsvwOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#pragma once
#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>
#include "Eigen/Dense"
#include "secure_matrix.h"


template < typename T >
class HungarianOptimizer
{
    static const int kHungarianOptimizerRowNotFound = -1;
    static const int kHungarianOptimizerColNotFound = -2;

  public:
    /// Setup the initial conditions for the algorithm.
    HungarianOptimizer();
    explicit HungarianOptimizer(const int max_optimization_size);
    ~HungarianOptimizer() {}

    SecureMat< T >* costs() { return &costs_; }

    T* costs(const size_t row, const size_t col) { return &(costs_(row, col)); }
    /**
     * @brief Find an assignment which maximizes the overall costs.
     * @param assignments
     * @return an array of pairs of integers. Each pair (i, j) corresponds to
     *         assigning agent i to task j.
     */
    void Maximize(std::vector< std::pair< size_t, size_t > >* assignments);
    /**
     * @brief Find an assignment which minimizes the overall costs.
     * @param assignments
     * @return return an array of pairs of integers. Each pair (i, j) corresponds to
     *         assinging agent i to task j.
     */
    void Minimize(std::vector< std::pair< size_t, size_t > >* assignments);
    /**
     * @brief init optimization TODO
     */
    void optimizationInit();
    /**
     * @brief clear optimization TODO
     */
    void optimizationClear();

  private:
    enum class Mark
    {
        kNone,
        kPrime,
        kStar
    };

    /**
     * Convert the final cost matrix into a set of assignments of agents to tasks.
     * Return the assignment in a vector of pair, the same as Minimize() and
     * Maximize()
     */
    void findAssignments(std::vector< std::pair< size_t, size_t > >* assignments);

    /// Is the cell (row, col) starred?
    bool isStarred(const size_t row, const size_t col) const { return marks_(row, col) == Mark::kStar; }

    /// Mark cell (row, col) with a star
    void Star(const size_t row, const size_t col)
    {
        marks_(row, col) = Mark::kStar;
        ++stars_in_col_[col];
    }

    /// Remove a star from cell (row, col)
    void Unstar(const size_t row, const size_t col)
    {
        marks_(row, col) = Mark::kNone;
        --stars_in_col_[col];
    }

    /**
     * Find a column in row 'row' containing a star, or return
     * kHungarianOptimizerColNotFound if no such column exists.
     */
    int findStarInRow(const size_t row) const;

    /**
     * Find a row in column 'col' containing a star, or return
     * kHungarianOptimizerRowNotFound if no such row exists.
     */
    int findStarInCol(const size_t col) const;

    /// Is cell (row, col) marked with a prime?
    bool isPrimed(const size_t row, const size_t col) const { return marks_(row, col) == Mark::kPrime; }

    /// Mark cell (row, col) with a prime.
    void Prime(const size_t row, const size_t col) { marks_(row, col) = Mark::kPrime; }

    /**
     * Find a column in row containing a prime, or return
     * kHungarianOptimizerColNotFound if no such column exists.
     */
    int findPrimeInRow(const size_t row) const;

    /// Remove the prime marks_ from every cell in the matrix.
    void clearPrimes();

    /// Does column col contain a star?
    bool colContainsStar(const size_t col) const { return stars_in_col_[col] > 0; }

    /// Is row 'row' covered?
    bool rowCovered(const size_t row) const { return rows_covered_[row]; }

    /// Cover row 'row'.
    void coverRow(const size_t row) { rows_covered_[row] = true; }

    /// Uncover row 'row'.
    void uncoverRow(const size_t row) { rows_covered_[row] = false; }

    /// Is column col covered?
    bool colCovered(const size_t col) const { return cols_covered_[col]; }

    /// Cover column col.
    void coverCol(const size_t col) { cols_covered_[col] = true; }

    /// Uncover column col.
    void uncoverCol(const size_t col) { cols_covered_[col] = false; }

    /// Uncover ever row and column in the matrix.
    void clearCovers();

    /// Find the smallest uncovered cell in the matrix.
    T findSmallestUncovered();

    /**
     * Find an uncovered zero and store its coordinates in (zeroRow_, zeroCol_)
     * and return true, or return false if no such cell exists.
     */
    bool findZero(size_t* zero_row, size_t* zero_col);

    /// Run the Munkres algorithm!
    void doMunkres();

    void checkStar();

    /**
     * Step 1:
     * For each row of the matrix, find the smallest element and subtract it from
     * every element in its row.  Go to Step 2.
     */
    void reduceRows();

    /**
     * Step 2:
     * Find a zero (Z) in the matrix. If there is no starred zero in its row or
     * column, star Z. Repeat for every element in the matrix. Go to Step 3.
     * Note: profiling shows this method to use 9.2% of the CPU - the next
     * slowest step takes 0.6%. It is hard to find a way further speed it up.
     */
    void starZeroes();

    /**
     * Step 3:
     * Cover each column containing a starred zero.  If all columns are covered,
     * the starred zeros describe a complete set of unique assignments.
     * In this case, terminate the algorithm.  Otherwise, go to Step 4.
     */
    void coverStarredZeroes();

    /**
     * Step 4:
     * Find a noncovered zero and prime it.  If there is no starred zero in the
     * row containing this primed zero, Go to Step 5.  Otherwise, cover this row
     * and uncover the column containing the starred zero. Continue in this manner
     * until there are no uncovered zeros left, then go to Step 6.
     */
    void primeZeroes();

    /**
     * Step 5:
     * Construct a series of alternating primed and starred zeros as follows.
     * Let Z0 represent the uncovered primed zero found in Step 4. Let Z1 denote
     * the starred zero in the column of Z0 (if any). Let Z2 denote the primed
     * zero in the row of Z1 (there will always be one). Continue until the
     * series terminates at a primed zero that has no starred zero in its column.
     * Unstar each starred zero of the series, star each primed zero of the
     * series, erase all primes and uncover every line in the matrix. Return to
     * Step 3.
     */
    void makeAugmentingPath();

    /**
     * Step 6:
     * Add the smallest uncovered value in the matrix to every element of each
     * covered row, and subtract it from every element of each uncovered column.
     * Return to Step 4 without altering any stars, primes, or covered lines.
     */
    void augmentPath();

    /// the max optimization size set to control memory
    int max_optimization_size_ = 1000;

    /// status of optimization initialization
    bool optimization_initialized_ = false;

    /// the size of the problem, i.e. std::max(#agents, #tasks).
    unsigned int matrix_size_ = 0;

    /// the expanded cost matrix.
    SecureMat< T > costs_;

    /// the greatest cost in the initial cost matrix.
    T max_cost_{0};

    /// which rows and columns are currently covered.
    std::vector< bool > rows_covered_;
    std::vector< bool > cols_covered_;

    /// the marks (star/prime/none) on each element of the cost matrix.
    SecureMat< Mark > marks_;

    /// the number of stars in each column - used to speed up coverStarredZeroes.
    std::vector< int > stars_in_col_;

    /// representation of a path_ through the matrix - used in Step 5.
    std::vector< std::pair< size_t, size_t > > assignments_;

    /// the locations of a zero found in Step 4.
    int zero_col_ = 0;
    int zero_row_ = 0;

    /// the width_ and height_ of the initial (non-expanded) cost matrix.
    unsigned int width_ = 0;
    unsigned int height_ = 0;

    /// The current state of the algorithm
    std::function< void() > fn_state_ = nullptr;

    std::vector< size_t > uncov_col_;
    std::vector< size_t > uncov_row_;
};

template < typename T >
HungarianOptimizer< T >::HungarianOptimizer() : HungarianOptimizer(1000)
{
}

template < typename T >
HungarianOptimizer< T >::HungarianOptimizer(const int max_optimization_size)
    : max_optimization_size_(max_optimization_size)
{
    costs_.reserve(max_optimization_size, max_optimization_size);
    stars_in_col_.reserve(max_optimization_size);
    rows_covered_.reserve(max_optimization_size);
    cols_covered_.reserve(max_optimization_size);
    assignments_.reserve(max_optimization_size);
    uncov_row_.reserve(max_optimization_size);
    uncov_col_.reserve(max_optimization_size);
}

/**
 * Find an assignment which maximizes the overall costs.
 * Return an array of pairs of integers. Each pair (i, j) corresponds to
 * assigning agent i to task j.
 */
template < typename T >
void HungarianOptimizer< T >::Maximize(std::vector< std::pair< size_t, size_t > >* assignments)
{
    optimizationInit();
    /**
     * operate maximizing problem as a minimizing one via substrating original
     * cost from max_cost_
     */
    for (size_t row = 0; row < height_; ++row)
    {
        for (size_t col = 0; col < width_; ++col)
        {
            costs_(row, col) = max_cost_ - costs_(row, col);
        }
    }
    Minimize(assignments);
}

/**
 * Find an assignment which minimizes the overall costs.
 * Return an array of pairs of integers. Each pair (i, j) corresponds to
 * assinging agent i to task j.
 */
template < typename T >
void HungarianOptimizer< T >::Minimize(std::vector< std::pair< size_t, size_t > >* assignments)
{
    optimizationInit();
    doMunkres();
    findAssignments(assignments);
    optimizationClear();
}

template < typename T >
void HungarianOptimizer< T >::optimizationInit()
{
    if (optimization_initialized_)
    {
        return;
    }
    width_ = static_cast< int >(costs_.width());
    if (width_ > 0)
    {
        height_ = static_cast< int >(costs_.height());
    }
    else
    {
        height_ = 0;
    }

    matrix_size_ = std::max(height_, width_);
    max_cost_ = 0;

    /**
     * generate the expanded cost matrix by adding extra 0s in order to make a
     * square matrix. Meanwhile, find the max cost in the matrix. It may be used
     * later, if we want to maximizing rather than minimizing the overall costs.
     */
    costs_.resize(matrix_size_, matrix_size_);
    for (size_t row = 0; row < matrix_size_; ++row)
    {
        for (size_t col = 0; col < matrix_size_; ++col)
        {
            if ((row >= height_) || (col >= width_))
            {
                costs_(row, col) = 0;
            }
            else
            {
                max_cost_ = std::max(max_cost_, costs_(row, col));
            }
        }
    }

    /// initially, none of the cells of the matrix are marked.
    marks_.resize(matrix_size_, matrix_size_);
    for (size_t row = 0; row < matrix_size_; ++row)
    {
        for (size_t col = 0; col < matrix_size_; ++col)
        {
            marks_(row, col) = Mark::kNone;
        }
    }

    stars_in_col_.assign(matrix_size_, 0);

    rows_covered_.assign(matrix_size_, false);
    cols_covered_.assign(matrix_size_, false);

    assignments_.resize(matrix_size_ * 2);

    optimization_initialized_ = true;
}

template < typename T >
void HungarianOptimizer< T >::optimizationClear()
{
    optimization_initialized_ = false;
}

/**
 * Convert the final costs matrix into a set of assignments of agents to tasks.
 * Return an array of pairs of integers, the same as the return values of
 * Minimize() and Maximize()
 */
template < typename T >
void HungarianOptimizer< T >::findAssignments(std::vector< std::pair< size_t, size_t > >* assignments)
{
    assignments->clear();
    for (size_t row = 0; row < height_; ++row)
    {
        for (size_t col = 0; col < width_; ++col)
        {
            if (isStarred(row, col))
            {
                assignments->push_back(std::make_pair(row, col));
                break;
            }
        }
    }
}

/**
 * Find a column in row 'row' containing a star, or return
 * kHungarianOptimizerColNotFound if no such column exists.
 */
template < typename T >
int HungarianOptimizer< T >::findStarInRow(const size_t row) const
{
    for (size_t col = 0; col < matrix_size_; ++col)
    {
        if (isStarred(row, col))
        {
            return static_cast< int >(col);
        }
    }

    return kHungarianOptimizerColNotFound;
}

/**
 * Find a row in column 'col' containing a star, or return
 * kHungarianOptimizerRowNotFound if no such row exists.
 */
template < typename T >
int HungarianOptimizer< T >::findStarInCol(const size_t col) const
{
    if (!colContainsStar(col))
    {
        return kHungarianOptimizerRowNotFound;
    }

    for (size_t row = 0; row < matrix_size_; ++row)
    {
        if (isStarred(row, col))
        {
            return static_cast< int >(row);
        }
    }

    /// NOT REACHED
    return kHungarianOptimizerRowNotFound;
}

/**
 * ind a column in row containing a prime, or return
 * kHungarianOptimizerColNotFound if no such column exists.
 */
template < typename T >
int HungarianOptimizer< T >::findPrimeInRow(const size_t row) const
{
    for (size_t col = 0; col < matrix_size_; ++col)
    {
        if (isPrimed(row, col))
        {
            return static_cast< int >(col);
        }
    }

    return kHungarianOptimizerColNotFound;
}

/// Remove the prime marks from every cell in the matrix.
template < typename T >
void HungarianOptimizer< T >::clearPrimes()
{
    for (size_t row = 0; row < matrix_size_; ++row)
    {
        for (size_t col = 0; col < matrix_size_; ++col)
        {
            if (isPrimed(row, col))
            {
                marks_(row, col) = Mark::kNone;
            }
        }
    }
}

/// Uncover every row and column in the matrix.
template < typename T >
void HungarianOptimizer< T >::clearCovers()
{
    for (size_t x = 0; x < matrix_size_; x++)
    {
        uncoverRow(x);
        uncoverCol(x);
    }
}

/// Find the smallest uncovered cell in the matrix.
template < typename T >
T HungarianOptimizer< T >::findSmallestUncovered()
{
    T minval = std::numeric_limits< T >::max();
    uncov_col_.clear();
    uncov_row_.clear();

    for (size_t i = 0; i < matrix_size_; ++i)
    {
        if (!rowCovered(i))
        {
            uncov_row_.push_back(i);
        }
        if (!colCovered(i))
        {
            uncov_col_.push_back(i);
        }
    }

    for (size_t row = 0; row < uncov_row_.size(); ++row)
    {
        for (size_t col = 0; col < uncov_col_.size(); ++col)
        {
            minval = std::min(minval, costs_(uncov_row_[row], uncov_col_[col]));
        }
    }

    return minval;
}

/**
 * Find an uncovered zero and store its coordinates in (zeroRow, zeroCol)
 * and return true, or return false if no such cell exists.
 */
template < typename T >
bool HungarianOptimizer< T >::findZero(size_t* zero_row, size_t* zero_col)
{
    uncov_col_.clear();
    uncov_row_.clear();

    for (unsigned int i = 0; i < matrix_size_; ++i)
    {
        if (!rowCovered(i))
        {
            uncov_row_.push_back(i);
        }
        if (!colCovered(i))
        {
            uncov_col_.push_back(i);
        }
    }
    if (uncov_row_.empty() || uncov_col_.empty())
    {
        return false;
    }

    for (size_t i = 0; i < uncov_row_.size(); ++i)
    {
        for (size_t j = 0; j < uncov_col_.size(); ++j)
        {
            if (costs_(uncov_row_[i], uncov_col_[j]) == 0)
            {
                *zero_row = uncov_row_[i];
                *zero_col = uncov_col_[j];
                return true;
            }
        }
    }
    return false;
}

/// Run the Munkres algorithm
template < typename T >
void HungarianOptimizer< T >::doMunkres()
{
    int max_num_iter = 1000;
    int num_iter = 0;
    fn_state_ = std::bind(&HungarianOptimizer::reduceRows, this);
    while (fn_state_ != nullptr && num_iter < max_num_iter)
    {
        fn_state_();
        ++num_iter;
    }
    if (num_iter >= max_num_iter)
    {
        checkStar();
    }
}

template < typename T >
void HungarianOptimizer< T >::checkStar()
{
    for (size_t row = 0; row < height_; ++row)
    {
        int star_col = -1;
        bool is_single = true;
        for (unsigned int col = 0; col < width_; ++col)
        {
            if (isStarred(row, col))
            {
                if (star_col == -1)
                {
                    star_col = col;
                }
                else
                {
                    is_single = false;
                    break;
                }
            }
        }
        if (!is_single)
        {
            for (unsigned int col = 0; col < width_; ++col)
            {
                Unstar(row, col);
            }
        }
    }
}

/**
 * Step 1:
 * For each row of the matrix, find the smallest element and substract it
 * from every element in its row. Then, go to Step 2.
 */
template < typename T >
void HungarianOptimizer< T >::reduceRows()
{
    for (size_t row = 0; row < matrix_size_; ++row)
    {
        T min_cost = costs_(row, 0);
        for (size_t col = 1; col < matrix_size_; ++col)
        {
            min_cost = std::min(min_cost, costs_(row, col));
        }
        for (size_t col = 0; col < matrix_size_; ++col)
        {
            costs_(row, col) -= min_cost;
        }
    }
    fn_state_ = std::bind(&HungarianOptimizer::starZeroes, this);
}

/**
 * Step 2:
 * Find a zero Z in the matrix. If there is no starred zero in its row
 * or column, star Z. Repeat for every element in the matrix. Then, go to
 * Step3.
 */
template < typename T >
void HungarianOptimizer< T >::starZeroes()
{
    /**
     * since no rows or columns are covered on entry to this step, we use the
     * covers as a quick way of making which rows & columns have stars in them
     */
    for (size_t row = 0; row < matrix_size_; ++row)
    {
        if (rowCovered(row))
        {
            continue;
        }
        for (size_t col = 0; col < matrix_size_; ++col)
        {
            if (colCovered(col))
            {
                continue;
            }
            if (costs_(row, col) == 0)
            {
                Star(row, col);
                coverRow(row);
                coverCol(col);
                break;
            }
        }
    }
    clearCovers();
    fn_state_ = std::bind(&HungarianOptimizer::coverStarredZeroes, this);
}

/**
 * Step 3:
 * Cover each column containing a starred zero. If all columns are covered,
 * the starred zeros describe a complete set of unique assignments. In this
 * case, terminate the algorithm. Otherwise, go to Step 4.
 */
template < typename T >
void HungarianOptimizer< T >::coverStarredZeroes()
{
    size_t num_covered = 0;

    for (size_t col = 0; col < matrix_size_; ++col)
    {
        if (colContainsStar(col))
        {
            coverCol(col);
            num_covered++;
        }
    }

    if (num_covered >= matrix_size_)
    {
        fn_state_ = nullptr;
        return;
    }
    fn_state_ = std::bind(&HungarianOptimizer::primeZeroes, this);
}

/**
 * Step 4:
 * Find a noncovered zero and prime it. If there is no starred zero in the
 * row containing this primed zero, go to Step 5. Otherwise, cover this row
 * and uncover the column containing the starred zero. Continue in this manner
 * until there are no uncovered zeros left, then go to Step 6.
 */
template < typename T >
void HungarianOptimizer< T >::primeZeroes()
{ /**
   * this loop is guaranteed to terminate in at most matrix_size_ iterations,
   * as findZero() returns a location only if there is at least one uncovered
   * zero in the matrix.  Each iteration, either one row is covered or the
   * loop terminates.  Since there are matrix_size_ rows, after that many
   * iterations there are no uncovered cells and hence no uncovered zeroes,
   * so the loop terminates.
   */
    for (;;)
    {
        size_t zero_row = 0;
        size_t zero_col = 0;
        if (!findZero(&zero_row, &zero_col))
        {
            /// No uncovered zeroes.
            fn_state_ = std::bind(&HungarianOptimizer::augmentPath, this);
            return;
        }

        Prime(zero_row, zero_col);
        /// size_t star_col = findStarInRow(zero_row);
        int star_col = findStarInRow(zero_row);

        if (star_col != kHungarianOptimizerColNotFound)
        {
            coverRow(zero_row);
            uncoverCol(star_col);
        }
        else
        {
            std::pair< size_t, size_t > first_assignment = std::make_pair(zero_row, zero_col);
            assignments_[0] = first_assignment;
            fn_state_ = std::bind(&HungarianOptimizer::makeAugmentingPath, this);
            return;
        }
    }
}

/**
 *  Step 5:
 * Construct a series of alternating primed and starred zeros as follows.
 * Let Z0 represent the uncovered primed zero found in Step 4.  Let Z1 denote
 * the starred zero in the column of Z0 (if any). Let Z2 denote the primed
 * zero in the row of Z1 (there will always be one).  Continue until the
 * series terminates at a primed zero that has no starred zero in its column.
 * unstar each starred zero of the series, star each primed zero of the
 * series, erase all primes and uncover every line in the matrix. Return to
 * Step 3.
 */
template < typename T >
void HungarianOptimizer< T >::makeAugmentingPath()
{
    bool done = false;
    size_t count = 0;

    /**
     * note: this loop is guaranteed to terminate within matrix_size_ iterations
     * because:
     * 1) on entry to this step, there is at least 1 column with no starred zero
     *    (otherwise we would have terminated the algorithm already.)
     * 2) each row containing a star also contains exactly one primed zero.
     * 4) each column contains at most one starred zero.
     *
     * Since the path_ we construct visits primed and starred zeroes alternately,
     * and terminates if we reach a primed zero in a column with no star, our
     * path_ must either contain matrix_size_ or fewer stars (in which case the
     * loop iterates fewer than matrix_size_ times), or it contains more.  In
     * that case, because (1) implies that there are fewer than matrix_size_
     * stars, we must have visited at least one star more than once. Consider
     * the first such star that we visit more than once; it must have been reached
     * immediately after visiting a prime in the same row.  By (2), this prime
     * is unique and so must have also been visited more than once.
     * Therefore, that prime must be in the same column as a star that has been
     * visited more than once, contradicting the assumption that we chose the
     * first multiply visited star, or it must be in the same column as more
     * than one star, contradicting (3). Therefore, we never visit any star
     * more than once and the loop terminates within matrix_size_ iterations.
     */

    while (!done)
    {
        /// first construct the alternating path...
        /// size_t row = findStarInCol(assignments_[count].second);
        int row = findStarInCol(assignments_[count].second);

        if (row != kHungarianOptimizerRowNotFound)
        {
            count++;
            assignments_[count].first = row;
            assignments_[count].second = assignments_[count - 1].second;
        }
        else
        {
            done = true;
        }

        if (!done)
        {
            size_t col = findPrimeInRow(assignments_[count].first);
            count++;
            assignments_[count].first = assignments_[count - 1].first;
            assignments_[count].second = col;
        }
    }

    /// then, modify it.
    for (size_t i = 0; i <= count; ++i)
    {
        size_t row = assignments_[i].first;
        size_t col = assignments_[i].second;

        if (isStarred(row, col))
        {
            Unstar(row, col);
        }
        else
        {
            Star(row, col);
        }
    }

    clearCovers();
    clearPrimes();
    fn_state_ = std::bind(&HungarianOptimizer::coverStarredZeroes, this);
}

/**
 * Step 6:
 * Add the smallest uncovered value in the matrix to every element of each
 * covered row, and subtract it from every element of each uncovered column.
 * Return to Step 4 without altering any stars, primes, or covered lines.
 */
template < typename T >
void HungarianOptimizer< T >::augmentPath()
{
    T minval = findSmallestUncovered();

    for (size_t row = 0; row < matrix_size_; ++row)
    {
        if (rowCovered(row))
        {
            for (size_t c = 0; c < matrix_size_; ++c)
            {
                costs_(row, c) += minval;
            }
        }
    }
    for (size_t col = 0; col < matrix_size_; ++col)
    {
        if (!colCovered(col))
        {
            for (size_t r = 0; r < matrix_size_; ++r)
            {
                costs_(r, col) -= minval;
            }
        }
    }
    fn_state_ = std::bind(&HungarianOptimizer::primeZeroes, this);
}



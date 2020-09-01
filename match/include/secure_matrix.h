
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

#include "Eigen/Dense"



template < typename T >
class SecureMat
{
  public:
    SecureMat() : height_(0), width_(0) { reserve(max_height_, max_width_); }
    /**
     * @brief: SecureMat height
     * @return: SecureMat height
     */
    size_t height() { return height_; }
    /**
     * @brief: SecureMat width
     * @return: SecureMat width
     */
    size_t width() { return width_; }

    /**
     * @brief: reserve memory of SecureMat
     * @params: reserve_height: height of reserve memory
     * @params: reserve_width: width of reserve memory
     */
    void reserve(const size_t reserve_height, const size_t reserve_width)
    {
        max_height_ = (reserve_height > max_height_) ? reserve_height : max_height_;
        max_width_ = (reserve_width > max_width_) ? reserve_width : max_width_;
        mat_.resize(max_height_, max_width_);
    }

    /**
     * @brief: resize memory of SecureMat
     * @params: resize_height: height of resize memory
     * @params: resize_width: width of resize memory
     */
    void resize(const size_t resize_height, const size_t resize_width)
    {
        height_ = resize_height;
        width_ = resize_width;
        if (resize_height <= max_height_ && resize_width <= max_width_)
        {
            return;
        }
        max_height_ = (resize_height > max_height_) ? resize_height : max_height_;
        max_width_ = (resize_width > max_width_) ? resize_width : max_width_;
        mat_.resize(max_height_, max_width_);
    }

    inline const T& operator()(const size_t row, const size_t col) const { return mat_(row, col); }

    inline T& operator()(const size_t row, const size_t col) { return mat_(row, col); }

  private:
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > mat_;
    size_t max_height_ = 1000;
    size_t max_width_ = 1000;
    size_t height_ = 0;
    size_t width_ = 0;
};


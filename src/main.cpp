#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace Eigen;

// ============================================================
// Data structures
// ============================================================

struct LineFeature
{
    double t;   // angle
    double d;   // distance to origin
    int s;      // sign {-1, +1}
};

// ============================================================
// Utility
// ============================================================

static constexpr double PI = 3.14159265358979323846;

double safeLog2(double x)
{
    return std::log(x) / std::log(2.0);
}

double xi(int x, int T)
{
    if (x == 0)
    {
        return 0.0;
    }
    return static_cast<double>(x) * safeLog2(static_cast<double>(x)) / static_cast<double>(T);
}

int signNoZero(double v)
{
    return (v >= 0.0) ? 1 : -1;
}

// ============================================================
// Random data generation
// ============================================================

MatrixXd generatePoints(int T, std::mt19937& rng)
{
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    MatrixXd x(T, 2);
    for (int i = 0; i < T; ++i)
    {
        x(i, 0) = dist(rng);
        x(i, 1) = dist(rng);
    }
    return x;
}

VectorXi generateLabels(const MatrixXd& x, double radius)
{
    const int T = static_cast<int>(x.rows());
    VectorXi y(T);

    for (int i = 0; i < T; ++i)
    {
        const double xp = x(i, 0);
        const double yp = x(i, 1);
        const double r = std::sqrt(xp * xp + yp * yp);
        y(i) = (r <= radius) ? 1 : 0;
    }

    return y;
}

vector<LineFeature> generateLineFeatures(int N, std::mt19937& rng)
{
    std::uniform_real_distribution<double> distT(0.0, PI);
    std::uniform_real_distribution<double> distD(-std::sqrt(2.0), std::sqrt(2.0));
    std::uniform_real_distribution<double> distS(-1.0, 1.0);

    vector<LineFeature> features;
    features.reserve(N);

    for (int i = 0; i < N; ++i)
    {
        LineFeature f;
        f.t = distT(rng);
        f.d = distD(rng);
        f.s = (distS(rng) >= 0.0) ? 1 : -1;
        features.push_back(f);
    }

    return features;
}

// ============================================================
// Entropy / cardinalities
// ============================================================

int card1(const VectorXi& x, int u)
{
    int count = 0;
    for (int i = 0; i < x.size(); ++i)
    {
        if (x(i) == u)
        {
            ++count;
        }
    }
    return count;
}

int card2(const VectorXi& x, const VectorXi& y, int u, int v)
{
    int count = 0;
    for (int i = 0; i < x.size(); ++i)
    {
        if (x(i) == u && y(i) == v)
        {
            ++count;
        }
    }
    return count;
}

int card3(const VectorXi& x, const VectorXi& y, const VectorXi& z, int u, int v, int w)
{
    int count = 0;
    for (int i = 0; i < x.size(); ++i)
    {
        if (x(i) == u && y(i) == v && z(i) == w)
        {
            ++count;
        }
    }
    return count;
}

double H1(const VectorXi& Y)
{
    const int T = static_cast<int>(Y.size());
    return safeLog2(static_cast<double>(T))
         - (xi(card1(Y, 0), T) + xi(card1(Y, 1), T));
}

double H2(const VectorXi& Y, const VectorXi& Xn)
{
    const int T = static_cast<int>(Y.size());
    return safeLog2(static_cast<double>(T))
         - (xi(card2(Y, Xn, 0, 0), T)
         +  xi(card2(Y, Xn, 0, 1), T)
         +  xi(card2(Y, Xn, 1, 0), T)
         +  xi(card2(Y, Xn, 1, 1), T));
}

double H3(const VectorXi& Y, const VectorXi& Xn, const VectorXi& Xm)
{
    const int T = static_cast<int>(Y.size());
    return safeLog2(static_cast<double>(T))
         - (xi(card3(Y, Xn, Xm, 0, 0, 0), T)
         +  xi(card3(Y, Xn, Xm, 0, 0, 1), T)
         +  xi(card3(Y, Xn, Xm, 0, 1, 0), T)
         +  xi(card3(Y, Xn, Xm, 0, 1, 1), T)
         +  xi(card3(Y, Xn, Xm, 1, 0, 0), T)
         +  xi(card3(Y, Xn, Xm, 1, 0, 1), T)
         +  xi(card3(Y, Xn, Xm, 1, 1, 0), T)
         +  xi(card3(Y, Xn, Xm, 1, 1, 1), T));
}

double mutInf(const VectorXi& Y, const VectorXi& Xn)
{
    return H1(Y) + H1(Xn) - H2(Y, Xn);
}

double condMutInf(const VectorXi& Y, const VectorXi& Xn, const VectorXi& Xm)
{
    return H2(Y, Xm) - H1(Xm) - H3(Y, Xn, Xm) + H2(Xn, Xm);
}

double Hp(const vector<double>& P)
{
    double sumP = std::accumulate(P.begin(), P.end(), 0.0);
    if (std::abs(sumP - 1.0) > 1000.0 * std::numeric_limits<double>::epsilon())
    {
        std::ostringstream oss;
        oss << "Sum of probabilities not equal to 1: " << sumP;
        throw std::runtime_error(oss.str());
    }

    double H = 0.0;
    for (double p : P)
    {
        if (p > std::numeric_limits<double>::epsilon() &&
            p < 1.0 - std::numeric_limits<double>::epsilon())
        {
            H += p * safeLog2(1.0 / p);
        }
    }
    return H;
}

// ============================================================
// Matrix construction
// ============================================================

MatrixXi buildMatrix(const MatrixXd& x, const vector<LineFeature>& f)
{
    const int T = static_cast<int>(x.rows());
    const int N = static_cast<int>(f.size());

    MatrixXi X = MatrixXi::Zero(T, N);

    for (int i = 0; i < T; ++i)
    {
        const double xp = x(i, 0);
        const double yp = x(i, 1);

        for (int j = 0; j < N; ++j)
        {
            const double tf = f[j].t;
            const double df = f[j].d;
            const int sf = f[j].s;

            const double value = -xp * std::sin(tf) + yp * std::cos(tf) - df;
            X(i, j) = (signNoZero(value) == sf) ? 1 : 0;
        }
    }

    return X;
}

VectorXi getColumn(const MatrixXi& X, int col)
{
    return X.col(col);
}

// ============================================================
// Global mutual information
// ============================================================

string buildRowKey(const MatrixXi& X, int row, const vector<int>& cols)
{
    std::ostringstream oss;
    for (size_t i = 0; i < cols.size(); ++i)
    {
        oss << X(row, cols[i]);
        if (i + 1 < cols.size())
        {
            oss << ',';
        }
    }
    return oss.str();
}

string buildRowKeyYX(const VectorXi& y, const MatrixXi& X, int row, const vector<int>& cols)
{
    std::ostringstream oss;
    oss << y(row) << '|';
    for (size_t i = 0; i < cols.size(); ++i)
    {
        oss << X(row, cols[i]);
        if (i + 1 < cols.size())
        {
            oss << ',';
        }
    }
    return oss.str();
}

double calcMi(const MatrixXi& X, const vector<int>& nu, int k, const VectorXi& y, double Hy, int T)
{
    vector<int> cols(nu.begin(), nu.begin() + k);

    map<string, int> countsX;
    map<string, int> countsYX;

    for (int i = 0; i < T; ++i)
    {
        countsX[buildRowKey(X, i, cols)]++;
        countsYX[buildRowKeyYX(y, X, i, cols)]++;
    }

    vector<double> Px;
    Px.reserve(countsX.size());
    for (const auto& kv : countsX)
    {
        Px.push_back(static_cast<double>(kv.second) / static_cast<double>(T));
    }

    vector<double> Pyx;
    Pyx.reserve(countsYX.size());
    for (const auto& kv : countsYX)
    {
        Pyx.push_back(static_cast<double>(kv.second) / static_cast<double>(T));
    }

    const double Hx = Hp(Px);
    const double Hyx = Hp(Pyx);

    return Hy + Hx - Hyx;
}

// ============================================================
// Console display
// ============================================================

void dispScores(const VectorXd& s, int k, const vector<int>& nu, double smax)
{
    cout << "---------";
    for (int n = 0; n < s.size(); ++n)
    {
        cout << "------";
    }
    cout << '\n';

    cout << "s  [" << k << "] = ";
    for (int n = 0; n < s.size(); ++n)
    {
        cout << fixed << setprecision(3) << s(n) << " ";
    }
    cout << '\n';

    cout << "-> max s(n) = " << fixed << setprecision(3) << smax
         << " - nu(" << k << ") = " << nu[k - 1] << '\n';
}

void dispCmi(int k, const MatrixXd& cmi)
{
    cout << "cmi[" << k << "] = ";
    for (int n = 0; n < cmi.cols(); ++n)
    {
        cout << fixed << setprecision(3) << cmi(k - 1, n) << " ";
    }
    cout << '\n';
}

// ============================================================
// OpenCV drawing helpers
// ============================================================

cv::Point2i worldToImage(double x, double y, int width, int height)
{
    int px = static_cast<int>((x + 1.0) * 0.5 * (width - 1));
    int py = static_cast<int>((1.0 - (y + 1.0) * 0.5) * (height - 1));
    return cv::Point2i(px, py);
}

cv::Scalar calcColor(int k, const VectorXd& smax)
{
    double sum = 0.0;
    for (int i = 0; i <= k; ++i)
    {
        sum += smax(i);
    }

    if (sum <= std::numeric_limits<double>::epsilon())
    {
        return cv::Scalar(0, 0, 0);
    }

    double currentPct = smax(k) / sum;
    double maxPct = 0.0;
    for (int i = 0; i <= k; ++i)
    {
        maxPct = std::max(maxPct, smax(i) / sum);
    }

    double red = (maxPct > 0.0) ? currentPct / maxPct : 0.0;
    red = std::clamp(red, 0.0, 1.0);

    // OpenCV uses BGR
    return cv::Scalar(0, 0, static_cast<int>(255.0 * red));
}

bool clipLineToBox(double t, double d, double xm, double xM, double ym, double yM,
                   cv::Point2d& p1, cv::Point2d& p2)
{
    vector<cv::Point2d> pts;

    const double st = std::sin(t);
    const double ct = std::cos(t);

    // Intersections with x = xm and x = xM
    if (std::abs(ct) > 1e-12)
    {
        double y1 = d / ct + xm * std::tan(t);
        if (y1 >= ym && y1 <= yM)
        {
            pts.emplace_back(xm, y1);
        }

        double y2 = d / ct + xM * std::tan(t);
        if (y2 >= ym && y2 <= yM)
        {
            pts.emplace_back(xM, y2);
        }
    }

    // Intersections with y = ym and y = yM
    if (std::abs(st) > 1e-12)
    {
        double x1 = ym / std::tan(t) - d / st;
        if (x1 >= xm && x1 <= xM)
        {
            pts.emplace_back(x1, ym);
        }

        double x2 = yM / std::tan(t) - d / st;
        if (x2 >= xm && x2 <= xM)
        {
            pts.emplace_back(x2, yM);
        }
    }

    // Remove duplicates
    vector<cv::Point2d> uniquePts;
    for (const auto& p : pts)
    {
        bool duplicate = false;
        for (const auto& q : uniquePts)
        {
            if (cv::norm(p - q) < 1e-8)
            {
                duplicate = true;
                break;
            }
        }
        if (!duplicate)
        {
            uniquePts.push_back(p);
        }
    }

    if (uniquePts.size() < 2)
    {
        return false;
    }

    p1 = uniquePts[0];
    p2 = uniquePts[1];
    return true;
}

void drawLineTri(cv::Mat& canvas,
                 double t, double d,
                 const cv::Scalar& color,
                 int lineWidth,
                 double xm, double xM, double ym, double yM)
{
    cv::Point2d a, b;
    if (!clipLineToBox(t, d, xm, xM, ym, yM, a, b))
    {
        return;
    }

    cv::Point2i p1 = worldToImage(a.x, a.y, canvas.cols, canvas.rows);
    cv::Point2i p2 = worldToImage(b.x, b.y, canvas.cols, canvas.rows);

    cv::line(canvas, p1, p2, color, lineWidth, cv::LINE_AA);
}

cv::Mat createPointsCanvas(const MatrixXd& x, const VectorXi& y, int width = 900, int height = 900)
{
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    // Border
    cv::rectangle(canvas, cv::Rect(0, 0, width - 1, height - 1), cv::Scalar(0, 0, 0), 1);

    for (int i = 0; i < x.rows(); ++i)
    {
        cv::Point2i p = worldToImage(x(i, 0), x(i, 1), width, height);
        cv::Scalar color = (y(i) == 1) ? cv::Scalar(0, 180, 0) : cv::Scalar(0, 0, 0);
        cv::circle(canvas, p, 2, color, cv::FILLED, cv::LINE_AA);
    }

    return canvas;
}

void showLines(cv::Mat& canvas, const vector<LineFeature>& f)
{
    for (const auto& line : f)
    {
        drawLineTri(canvas, line.t, line.d, cv::Scalar(255, 0, 0), 1, -1.0, 1.0, -1.0, 1.0);
    }
}

void drawBarChart(cv::Mat& img,
                  const vector<double>& values,
                  double maxVal,
                  const string& title,
                  const cv::Scalar& color,
                  const cv::Rect& area)
{
    cv::rectangle(img, area, cv::Scalar(240, 240, 240), cv::FILLED);
    cv::rectangle(img, area, cv::Scalar(0, 0, 0), 1);

    cv::putText(img, title, cv::Point(area.x + 10, area.y + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    if (values.empty() || maxVal <= 0.0)
    {
        return;
    }

    const int left = area.x + 40;
    const int right = area.x + area.width - 20;
    const int top = area.y + 40;
    const int bottom = area.y + area.height - 35;

    cv::line(img, cv::Point(left, bottom), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 1);
    cv::line(img, cv::Point(left, top), cv::Point(left, bottom), cv::Scalar(0, 0, 0), 1);

    const int count = static_cast<int>(values.size());
    const double step = static_cast<double>(right - left) / std::max(count, 1);
    const int barWidth = std::max(4, static_cast<int>(0.7 * step));

    for (int i = 0; i < count; ++i)
    {
        double ratio = values[i] / maxVal;
        ratio = std::clamp(ratio, 0.0, 1.0);

        int x = left + static_cast<int>(i * step + 0.15 * step);
        int h = static_cast<int>(ratio * (bottom - top));
        int y = bottom - h;

        cv::rectangle(img, cv::Rect(x, y, barWidth, h), color, cv::FILLED);

        cv::putText(img, std::to_string(i + 1), cv::Point(x, bottom + 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
}

cv::Mat createCmiCanvas(const VectorXd& smax, const VectorXd& mi, double Hy, int currentK)
{
    cv::Mat canvas(800, 1000, CV_8UC3, cv::Scalar(255, 255, 255));

    vector<double> svals(currentK), mvals(currentK);
    double maxS = 0.0;
    double maxM = Hy;

    for (int i = 0; i < currentK; ++i)
    {
        svals[i] = smax(i);
        mvals[i] = mi(i);
        maxS = std::max(maxS, svals[i]);
        maxM = std::max(maxM, mvals[i]);
    }

    drawBarChart(canvas, svals, std::max(maxS, 1e-12),
                 "Conditional Mutual Information of selected features",
                 cv::Scalar(0, 0, 255),
                 cv::Rect(40, 40, 920, 300));

    drawBarChart(canvas, mvals, std::max(maxM, 1e-12),
                 "Global Mutual Information I(Y; {Xk}) evolution",
                 cv::Scalar(255, 0, 0),
                 cv::Rect(40, 420, 920, 300));

    // Horizontal line at Hy in bottom chart
    const cv::Rect area(40, 420, 920, 300);
    const int left = area.x + 40;
    const int right = area.x + area.width - 20;
    const int top = area.y + 40;
    const int bottom = area.y + area.height - 35;

    double ratio = Hy / std::max(maxM, 1e-12);
    ratio = std::clamp(ratio, 0.0, 1.0);
    int yLine = bottom - static_cast<int>(ratio * (bottom - top));

    cv::line(canvas, cv::Point(left, yLine), cv::Point(right, yLine), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    cv::putText(canvas, "Hy", cv::Point(right - 30, yLine - 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

    return canvas;
}

// ============================================================
// Main translated program
// ============================================================

int main()
{
    try
    {
        // Parameters
        const double R = 0.5;
        const int T = 5000;
        const int N = 500;
        const int K = 15;

        // Storage
        VectorXd s = VectorXd::Zero(N);
        MatrixXd cmi = MatrixXd::Zero(K, N);
        vector<int> nu(K, -1);
        VectorXd smax = VectorXd::Zero(K);
        VectorXd mi = VectorXd::Zero(K);

        // Random generator
        std::random_device rd;
        std::mt19937 rng(rd());

        // Generate data
        MatrixXd x = generatePoints(T, rng);
        VectorXi y = generateLabels(x, R);
        vector<LineFeature> fl = generateLineFeatures(N, rng);

        // Initial display
        cv::Mat pointsCanvas = createPointsCanvas(x, y, 900, 900);
        showLines(pointsCanvas, fl);
        cv::imshow("Figure 1 - Points and line features", pointsCanvas);

        // Build boolean matrix X (stored as 0/1 integers)
        MatrixXi X = buildMatrix(x, fl);
        const double Hy = H1(y);

        // Initial mutual information scores
        for (int n = 0; n < N; ++n)
        {
            s(n) = mutInf(y, getColumn(X, n));
        }

        // Main selection loop
        for (int k = 0; k < K; ++k)
        {
            // Select best informative feature
            Eigen::Index bestIdx;
            smax(k) = s.maxCoeff(&bestIdx);
            nu[k] = static_cast<int>(bestIdx);

            dispScores(s, k + 1, nu, smax(k));

            mi(k) = calcMi(X, nu, k + 1, y, Hy, T);

            cv::Mat cmiCanvas = createCmiCanvas(smax, mi, Hy, k + 1);
            cv::imshow("Figure 2 - CMI and Global MI", cmiCanvas);

            // Draw selected line
            cv::Scalar color = calcColor(k, smax);
            drawLineTri(pointsCanvas, fl[nu[k]].t, fl[nu[k]].d, color, 2, -1.0, 1.0, -1.0, 1.0);
            cv::imshow("Figure 1 - Points and line features", pointsCanvas);

            // Update scores with conditional mutual information
            VectorXi Xnu = getColumn(X, nu[k]);
            for (int n = 0; n < N; ++n)
            {
                cmi(k, n) = condMutInf(y, getColumn(X, n), Xnu);
                s(n) = std::min(s(n), cmi(k, n));
            }

            dispCmi(k + 1, cmi);

            // Small refresh
            cv::waitKey(20);
        }

        cout << "\nPress any key to continue...\n";
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

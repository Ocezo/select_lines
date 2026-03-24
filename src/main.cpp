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

struct ProgramOptions
{
    bool gui = true;
    unsigned int seed = std::random_device{}();
};

ProgramOptions parseArgs(int argc, char* argv[])
{
    ProgramOptions options;

    for (int i = 1; i < argc; ++i)
    {
        const string arg = argv[i];

        if (arg == "--no-gui")
        {
            options.gui = false;
        }
        else if (arg.rfind("--seed=", 0) == 0)
        {
            options.seed = static_cast<unsigned int>(stoul(arg.substr(7)));
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return options;
}

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

template <typename Derived>
double H1(const Eigen::MatrixBase<Derived>& Y)
{
    const int T = static_cast<int>(Y.size());
    int counts[2] = {0, 0};

    for (int i = 0; i < Y.size(); ++i)
    {
        ++counts[Y(i)];
    }

    return safeLog2(static_cast<double>(T))
         - (xi(counts[0], T) + xi(counts[1], T));
}

template <typename DerivedY, typename DerivedX>
double H2(const Eigen::MatrixBase<DerivedY>& Y, const Eigen::MatrixBase<DerivedX>& Xn)
{
    const int T = static_cast<int>(Y.size());
    int counts[2][2] = {{0, 0}, {0, 0}};

    for (int i = 0; i < Y.size(); ++i)
    {
        ++counts[Y(i)][Xn(i)];
    }

    return safeLog2(static_cast<double>(T))
         - (xi(counts[0][0], T)
         +  xi(counts[0][1], T)
         +  xi(counts[1][0], T)
         +  xi(counts[1][1], T));
}

template <typename DerivedY, typename DerivedX, typename DerivedZ>
double H3(const Eigen::MatrixBase<DerivedY>& Y, const Eigen::MatrixBase<DerivedX>& Xn, const Eigen::MatrixBase<DerivedZ>& Xm)
{
    const int T = static_cast<int>(Y.size());
    int counts[2][2][2] = {};

    for (int i = 0; i < Y.size(); ++i)
    {
        ++counts[Y(i)][Xn(i)][Xm(i)];
    }

    double sum = 0.0;
    for (int a = 0; a < 2; ++a)
    {
        for (int b = 0; b < 2; ++b)
        {
            for (int c = 0; c < 2; ++c)
            {
                sum += xi(counts[a][b][c], T);
            }
        }
    }

    return safeLog2(static_cast<double>(T)) - sum;
}

template <typename DerivedY, typename DerivedX, typename DerivedZ>
double condMutInfFast(const Eigen::MatrixBase<DerivedY>& Y,
                      const Eigen::MatrixBase<DerivedX>& Xn,
                      const Eigen::MatrixBase<DerivedZ>& Xm,
                      double selectedFeatureBaseCmi)
{
    const int T = static_cast<int>(Y.size());
    int counts2[2][2] = {{0, 0}, {0, 0}};
    int counts3[2][2][2] = {};

    for (int i = 0; i < T; ++i)
    {
        const int xn = Xn(i);
        const int xm = Xm(i);
        ++counts2[xn][xm];
        ++counts3[Y(i)][xn][xm];
    }

    double sum2 = 0.0;
    for (int a = 0; a < 2; ++a)
    {
        for (int b = 0; b < 2; ++b)
        {
            sum2 += xi(counts2[a][b], T);
        }
    }

    double sum3 = 0.0;
    for (int a = 0; a < 2; ++a)
    {
        for (int b = 0; b < 2; ++b)
        {
            for (int c = 0; c < 2; ++c)
            {
                sum3 += xi(counts3[a][b][c], T);
            }
        }
    }

    const double logT = safeLog2(static_cast<double>(T));
    const double h2xnxm = logT - sum2;
    const double h3yxnxm = logT - sum3;

    return selectedFeatureBaseCmi - h3yxnxm + h2xnxm;
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

// ============================================================
// Global mutual information
// ============================================================

double calcMi(const MatrixXi& X, const vector<int>& nu, int k, const VectorXi& y, double Hy, int T)
{
    const int stateCountX = 1 << k;
    const int stateCountYX = 1 << (k + 1);

    vector<int> countsX(stateCountX, 0);
    vector<int> countsYX(stateCountYX, 0);

    for (int row = 0; row < T; ++row)
    {
        int xState = 0;
        for (int idx = 0; idx < k; ++idx)
        {
            xState = (xState << 1) | X(row, nu[idx]);
        }

        ++countsX[xState];
        ++countsYX[(y(row) << k) | xState];
    }

    double sumX = 0.0;
    for (int count : countsX)
    {
        sumX += xi(count, T);
    }

    double sumYX = 0.0;
    for (int count : countsYX)
    {
        sumYX += xi(count, T);
    }

    const double logT = safeLog2(static_cast<double>(T));
    const double Hx = logT - sumX;
    const double Hyx = logT - sumYX;

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

int main(int argc, char* argv[])
{
    const ProgramOptions options = parseArgs(argc, argv);
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
    VectorXd columnEntropy = VectorXd::Zero(N);

    // Random generator
    std::mt19937 rng(options.seed);

    // Generate data
    MatrixXd x = generatePoints(T, rng);
    VectorXi y = generateLabels(x, R);
    vector<LineFeature> fl = generateLineFeatures(N, rng);

    cout << "Seed: " << options.seed << "\n";

    // Initial display
    cv::Mat pointsCanvas;
    if (options.gui)
    {
        cv::namedWindow("Figure 1 - Points and line features", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Figure 2 - CMI and Global MI", cv::WINDOW_AUTOSIZE);

        pointsCanvas = createPointsCanvas(x, y, 800, 800);
        showLines(pointsCanvas, fl);
        cv::imshow("Figure 1 - Points and line features", pointsCanvas);
    }

    // Build boolean matrix X (stored as 0/1 integers)
    MatrixXi X = buildMatrix(x, fl);
    const double Hy = H1(y);

    // Initial mutual information scores
    for (int n = 0; n < N; ++n)
    {
        const auto Xn = X.col(n);
        columnEntropy(n) = H1(Xn);
        s(n) = Hy + columnEntropy(n) - H2(y, Xn);
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

        if (options.gui)
        {
            cv::Mat cmiCanvas = createCmiCanvas(smax, mi, Hy, k + 1);
            cv::moveWindow("Figure 2 - CMI and Global MI", 1000, 100);
            cv::imshow("Figure 2 - CMI and Global MI", cmiCanvas);

            // Draw selected line
            cv::Scalar color = calcColor(k, smax);
            drawLineTri(pointsCanvas, fl[nu[k]].t, fl[nu[k]].d, color, 2, -1.0, 1.0, -1.0, 1.0);
            cv::moveWindow("Figure 1 - Points and line features", 100, 100);
            cv::imshow("Figure 1 - Points and line features", pointsCanvas);
        }

        // Update scores with conditional mutual information
        const auto Xnu = X.col(nu[k]);
        const double selectedFeatureBaseCmi = H2(y, Xnu) - columnEntropy(nu[k]);
        for (int n = 0; n < N; ++n)
        {
            const auto Xn = X.col(n);
            cmi(k, n) = condMutInfFast(y, Xn, Xnu, selectedFeatureBaseCmi);
            s(n) = std::min(s(n), cmi(k, n));
        }

        dispCmi(k + 1, cmi);

        if (options.gui)
        {
            // Small refresh
            cv::waitKey(20);
        }
    }

    if (options.gui)
    {
        cout << "\nPress any key to continue...\n";
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return EXIT_SUCCESS;
}

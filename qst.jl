using LinearAlgebra
using SparseArrays
using Printf

function unif(Q::SparseMatrixCSC; ufact = 1.01)::Tuple{Float64,SparseMatrixCSC}
    qv = maximum(abs.(diag(Q))) * ufact
    P = Q / qv + spdiagm(0 => ones(Q.n))
    (qv, P)
end

function gsstep!(x::Array{Float64,1}, Q::SparseMatrixCSC, b::Array{Float64,1};
    alpha::Float64=1.0, sigma::Float64=0.0, omega::Float64=1.0)
    for j in 1:Q.n
        tmpd = 0.0
        tmpx = b[j] / alpha
            for z in Q.colptr[j]:(Q.colptr[j+1]-1)
            i = Q.rowval[z]
            if i == j
                tmpd = Q.nzval[z]
                tmpx += sigma * x[j]
            else
                tmpx -= Q.nzval[z] * x[i]
            end
        end
        x[j] = omega / tmpd * tmpx + (1.0 - omega) * x[j]
    end
end

function stgs(Q::SparseMatrixCSC;
    x0::Array{Float64,1}=fill(1/Q.n, Q.n),
    maxiter=5000, steps=50, rtol=1.0e-6)::Array{Float64,1}
    b = zeros(Q.n)
    x = x0
    iter = 0
    conv = false
    rerror = 0.0
    while true
        prevx = x
        for i in 1:steps
            gsstep!(x, Q, b)
            x /= sum(x)
        end
        rerror = maximum(abs.((x - prevx) ./ x))
        iter += steps
        if rerror < rtol
            conv = true
            break
        end
        if iter >= maxiter
            break
        end
    end
    @printf "convergence   : %s\n" conv ? "true" : "false"
    @printf "iteration     : %d / %d\n" iter maxiter
    @printf "relative error: %e < %e\n" rerror rtol
    x
end

function stpower(Q::SparseMatrixCSC;
    x0::Array{Float64,1}=fill(1.0/Q.n, Q.n),
    ufact=1.01, maxiter=5000, steps=50, rtol=1.0e-6)::Array{Float64,1}
    (qv, P) = unif(Q, ufact=ufact)
    x = x0
    iter = 0
    conv = false
    rerror = 0.0
    while true
        prevx = x
        for i in 1:steps
            x = P' * x
            x /= sum(x)
        end
        rerror = maximum(abs.((x - prevx) ./ x))
        iter += steps
        if rerror < rtol
            conv = true
            break
        end
        if iter >= maxiter
            break
        end
    end
    @printf "convergence   : %s\n" conv ? "true" : "false"
    @printf "iteration     : %d / %d\n" iter maxiter
    @printf "relative error: %e < %e\n" rerror rtol
    x
end

function qstgs(Q::SparseMatrixCSC, xi::Array{Float64,1};
    x0::Array{Float64,1}=fill(1/Q.n, Q.n),
    maxiter=5000, steps=50, rtol=1.0e-6)::Tuple{Float64,Array{Float64,1}}
    b = zeros(Q.n)
    x = x0
    iter = 0
    conv = false
    rerror = 0.0
    gam = 0.0
    while true
        prevx = x
        for i in 1:steps
            gam = dot(x, xi)
            gsstep!(x, Q, b, sigma=-gam)
            x /= sum(x)
        end
        rerror = maximum(abs.((x - prevx) ./ x))
        iter += steps
        if rerror < rtol
            conv = true
            break
        end
        if iter >= maxiter
            break
        end
    end
    @printf "convergence   : %s\n" conv ? "true" : "false"
    @printf "iteration     : %d / %d\n" iter maxiter
    @printf "relative error: %e < %e\n" rerror rtol
    gam, x
end

# template <typename T1, typename T2, typename MatT>
# void ctmc_st_gth(const T1& Q, T2& x, MatT) {
#   const int n = nrow(Q, MatT());
#   dense_matrix A(n,n);
#   dcopy(Q, A, MatT(), DenseMatrixT());

#   using traits1 = vector_traits<T2>;
#   double* valueX = traits1::value(x);
#   const int incx = traits1::inc(x);

#   using traits2 = dense_matrix_traits<dense_matrix>;
#   double* valueA = traits2::value(A);
#   const int lda = traits2::ld(A);

#   for (int l=n; l>=2; l--) {
#     // tmp = sum(A(l,1:l-1))
#     double tmp = 0.0;
#     for (int j=1; j<=l-1; j++) {
#       tmp += elemA(l,j,valueA,lda);
#     }
#     for (int j=1; j<=l-1; j++) {
#       for (int i=1; i<=l-1; i++) {
#         if (i != j) {
#           elemA(i,j,valueA,lda) += elemA(l,j,valueA,lda) * elemA(i,l,valueA,lda) / tmp;
#         }
#       }
#     }
#     for (int i=1; i<=l-1; i++) {
#       elemA(i,l,valueA,lda) /= tmp;
#     }
#     for (int j=1; j<=l-1; j++) {
#       elemA(l,j,valueA,lda) = 0.0;
#     }
#     elemA(l,l,valueA,lda) = -1;
#   }

#   double total = 0.0;
#   double* tmpX = valueX;
#   *tmpX = 1.0;
#   total += *tmpX;
#   tmpX += incx;
#   for (int l=2; l<=n; l++, tmpX+=incx) {
#     *tmpX = 0.0;
#     double* Xi = valueX;
#     for (int i=1; i<=l-1; i++, Xi+=incx) {
#       *tmpX += *Xi * elemA(i,l,valueA,lda);
#     }
#     total += *tmpX;
#   }
#   dscal(1.0/total, x);
# }


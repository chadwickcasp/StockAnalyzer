import Data.List

-- main = print(sum_five_three_multiples 1000)
main = print (fib_up_to 4000000)
-- main = print()
-- main = print(1000 `mod` 3)
-- multiples_up_to 1 1 = [1]
-- multiples_up_to 1 n = [1, n]

-- multiples_up_to m n = 
--     if n == 1 
--         then [1]
--         else if n == m then [1, m]
--             else if n `mod` m /= 0
--                 then multiples_up_to m n-1
--                 else multiples_up_to m (n/m) ++ [n]

fib_up_to :: (Integral a) => a -> a -> [a]
fib_up_to m
    | m == 3 = [1, 2, 3]
    | last fib_up_to n >= m = fib_up_to n-1
    | otherwise = [last fib_up_to m-2] ++ [last fib_up_to m-1] ++ fib_up_to m-2



-- three_multiples 333 = 999



-- five_three_multiples :: Eq a => [a] -> [a]
-- i = 3 `mod` 3

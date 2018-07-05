module StateMonad
( 
   Calculator,
   
)
where

import qualified Data.Strings as S
import Control.Monad.State.Lazy

type Calculator a = State a (a -> a)  -- State ouput state


(~~) :: (a -> a) -> Calculator a
(~~) f = state $ \s -> (f, f s)

(~@) :: (a -> a -> a) -> a -> Calculator a
(~@) op x = (~~) (op x)

(~+) :: (Num a) => a -> Calculator a
(~+) x = (~~) (+x)

(~-) :: (Num a) => a -> Calculator a
(~-) x = (~@) (-) x

(~^) :: (Integral a, Num b) => a -> Calculator b
(~^) x = (~~) (\y -> y ^ x)

kpow :: ((a -> a) -> Calculator a) -> Integer -> ((a -> a) -> Calculator a)
-- Kleisli power operator
kpow f n | n == 0 = return
         | n == 1 = f
         | otherwise = (f >=> (kpow f (n-1)))


op :: (Num a) => Calculator a
op = do
    (~+) 10
    (~-) 1
    (~~) (*2)
    (~^) 2
    >>=
    (kpow (~~) 0)


main :: IO ()
main = do 
    let (_, res) = runState op 0 in putStrLn ("WoW " ++ (show res))

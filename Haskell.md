# Haskell

[TOC]

假设读者都有函数式编程方面的知识.

## 概念

变量、表达式、类型、绑定、作用域、模式

## 基本语法

只写一些容易遗忘和另类的语法。

### 注释

```haskell
-- comments
{- comments
multi-lines
-}
```



 ### 表达式

```haskell
3::[2, 4]  -- ==[3,2,4]
case x of 'a' -> 1
          'b' -> 2
          'c' -> 3
          
if ... then ... else ...

let x=a in expr
expr where x=a

sgn :: Double -> Double
sgn x | x<0 = -1
      | x>0 = 1
      | x==0 = 0
```



### 类型

```haskell
f :: Double -> Double
n :: Int
x :: Double
x :: String/Char/Num...

5 :: Num a => a  -- environment of types
f x = x+1 :: Num a => a -> a
```



### 模块

```haskell
module Main (
binding,
module YourModule,
DataType(Constructor1,...),
ClassDef(classMethod1,...),
...
) where  -- the name of the module
import Data.List   -- import an exterial module
import MyModule -- MyModule.foo
import MyModule.SubModule (MyType, myvariable,...) [hiding (...)] -- import some, hind some
import qualified YourModule as YM   -- YM.foo

... -- expressions, the body of the module
binding
DataType
...

main :: IO()     -- like main function in C and __main__ in python
main = print(1+2)
```



#### 注.

* Haskell 默认导入Prelude模块
* List 元素类型必须一样



### ghci交互

```haskell
:t x    -- type of x
:info x  -- information of x
:l m -- load a module
```



### 运算符

```haskell
++ -- concatenate two lists
[1,2,3]!!2 -- 3
fst, snd, head/tail, init/last -- get element(s) in a list
null [] -- is it an empty list?
take, drop, reverse, and/or, elem/notElem -- operate lists
/= -- not equal
```



## 数据类型

### 模式

```haskell
data Position = Position Double Double
data Person = Person String Int Bool

older :: Person -> Person -> Bool
older Person(a1, b1, c1) Person(a2, b2, c2) = b1 > b2
match Person(a1, b1, c1) Person(a2, b2, c2) = c1 \= c2 && abs(b1-b2)<=2

older p1@Person(a1, b1, c1) p2@Person(a2, b2, c2) = b1 > b2  -- @pattern

data Nat = Zero | Succ Nat -- Zero, Succ: constructor of Nat
```



### 记录语法

```haskell
data Person = Person {getName:: String, getAge:: Int, getGender:: Bool}
p = Person {getName='', getAge=0, getGender=True}
```



```haskell
-- exercise
data Person = Person {getName:: String, getAge:: Int, getGender:: Bool}

show_info :: Person -> String
show_info p = "My name is " ++ (getName p) ++ ". I am " ++ (show (getAge p)) ++ " years old."

main = do
   print (show_info p) where
   p = Person {getName="William", getAge=30, getGender=True}
```



## 列表递归

列表的盒子比喻

列表元素类型统一

```haskell
data [a] = a: [a] | []
1:2:[] == [1,2]
[1,3..7]
head (x:xs) = x  -- list pattern

repeat :: a -> [a]
repeat x = x:repeat x
```

### 列表操作

```haskell
map
filter
foldl/foldr
-- foldl f a [b, c] == f(f(a,b),c)
scanl/scanr
```





## 元组，高阶函数

### 元组

元组元素可以有不同类型

```haskell
(a,b,c)
("Haskell", 1990, True) :: (String, Int, Bool)
```



### 高阶函数

（反）Curry化

```haskell
zip/unzip

zipWith (+) [1,2] [1,3]
-- [2,5]

$ -- application
& -- piple
f $ x == f x == x & f
\x -> f x == f   -- dummy function
```



## 类型类

class 定义类型类

instance 定义类型类的实例

```haskell
Eq -- typeclass
class Eq a where -- declaration of Eq
    (==) :: a -> a -> Bool
    (/=) :: a -> a -> Bool

instance Eq Person where
    (==) :: Person -> Person -> Bool
    p == q = getName p == getName q
    
Show/Read
```

### 数字类型类

```haskell
-- classes related to numbers
Real <= Ord, Num <= Eq
Enum
Bounded
-- number class
RealFloat <= Floating, RealFrac <= Fractional <= Num <= (Eq, Show) -- Num a => Fractional a
Floating <= Integral <= (Real, Enum) <= Num
RealFrac <= Real <= (Num, Ord)
```



注. Haskell 类型约束信息并不会保存到数据中



| Haskell        | 数据    | 类型                         | 类型类           |
| -------------- | ------- | ---------------------------- | ---------------- |
| 范畴论         | 元素    | 对象/集合                    | 子范畴/范畴      |
| 例子           | 3       | Int                          | Num/Show/Eq/Hack |
| oop            | 对象    | 类型                         | 元类             |
| 代数学         | 元素    | 代数系统                     | 代数系统类型     |
| Haskell 关键词 | 3:: Int | instance,data, type, newtype | class            |



## 类型声明

### 类型别名、新类型声明

```haskell
type List a = [a] -- 类型别名, 右侧没有函数
type Function1D = Double -> Double
type Function1D a = a -> a

-- newtype 只接收一个参数
newtype Transform a = Transform （a->a） -- 接近 data (看作 data 特例) 而非 type，比data快速
newtype Const a b = Const a
newtype State s a = State {runState :: s -> (a, s)}
```

newtype 避免额外的打包解包过程, 右侧包裹的类型必须是上层类型.



## 惰性求值

### 概念

1. 任务盒(thunk): 只是一个表达式，不进行求值
2. 常态(normal form): 求值后的表达式; 弱常态：部分求值, 所有构造函数创建的数据都是若常态
3. 底(bottom): $\perp$

```haskell
:print  -- lookup the evaluation
:sprint
:force
```



```haskell
-- 计算到若常态
seq :: a -> b -> a
$!
-- 计算到常态
deepseq
$!!
force :: NFData a => a -> a
```



## 透镜组: 构造函数到构造函数的映射

```haskell
type Lens b a = Functor f => (a -> f a) -> b -> f b  -- lens
```

$Lens(b,a) = Hom_{Set}(Hom_E(a,Fa), Hom_E(b, Fb)), F:C\to D\subset E$ 透镜映射

$lens(f, p)=F(s(p))(f(g(p))):Fb, p:b,g:b\to a, s:a\times b\to b, s(p):a\to b$

esp. $s(g(p),p) =1_b(p), g(s(x,p))=1_a(x)​$. $s​$: setter, $g​$:getter

```haskell
-- Lens
data Point = Point {getX:: Double, getY:: Double}
data Line = Line {getStart:: Point, getEnd:: Point}
let setX x p = p {getX = x} -- setX x (P _ _) = (P x _) 
let xlens f p = fmap (\x -> setX x p) $ f (getX p)

--
xlens f p = fmap setter $ f $ getter p where
    setter :: Double -> Point
    setter x = p {getX=x}
--
xlens f p = fmap (setter p) $ f $ getter p

-- over
over :: Functor f => ((a -> f a) -> b -> f b) -> (a -> a) -> b -> b
over :: ((a -> Identity a) -> b -> Identity b) -> (a -> a) -> b -> b
over lens f x = runIdentity $ lifted x
    where
        lifted = lens (Identity . f)
-- over
over lens = \f -> runIdentity . (lens (Identity . f)) :: (a -> a) -> b -> b
-- over
xlens f p = (setter p) $ f $ getter p

-- set
set xlens x p == over xlens (const x) p

-- view
view :: Lens b a -> b -> a
view lens x = getConst ((lens Const) x) -- view lens = getConst . (lens Const)
-- view lens == getter

x ^. lens = view lens x
lens %~ f x = over lens f x
lens .~ a x = set lens a x
```



$over (f)=d(lens(c(f)))$, $c:a\to F(a)$-constructor, $d:F(b)\to b$-deconstructor

$set(x)=over(.\to x)=s(x,\cdot)$.



## 应用函子

构造映射$\eta: I\dot\to T$, $T:X\to X$.

$\eta$满，则称为构造函数.

```haskell
class Functor f => Applicative f where
    pure :: a -> f a   -- minimum context
    (<*>) :: f (a -> b) -> f a -> f b
    pure id <*> v = v  -- identity
    pure (.) <*> u <*> v <*> w = u <*> (v <*> w) -- compostition
    pure f <*> pure x = pure (f x)  -- homomorphism
    u <*> pure y = pure ($ y) <*> u -- interchange
```

$\circledast: F(a\to b)\to Fa\to Fb, a,b,a\to b:Hack, F:Fun(Hack)$,



#### Fact

$\eta(f)\circledast x=F(f)(x), f\circledast \eta(x)=\eta(e_x)\circledast f$

$\eta(f)\circledast \eta(x)=\eta(f(x))$, $\eta_a:a\to Fa$: add minimum context



### 自然升格

```haskell
<$> :: (a -> b) -> f a -> f b
f <$> x = fmap f x  -- (= (pure f) <*> x)
<$ :: b -> f a -> f b
<$ = fmap . const  -- a <$ x = fmap (const a) x, x $> a

*> :: Applicative f => f a -> f b -> f b
x *> y = (id <$ x) <*> y = (\_ -> id) <$> x <*> y -- y <* x
{-
f a = c -> a, -- F = Hom(c, .)
x *> y = y

f a = [a]
x *> y = y ++ y ++ ... ++ y
-}
-- example
replicate <$> Just 1 *> Just (+1) <*> Just 1234
-- > 

-- listAx
```

### List、Reader 应用函子

```haskell
f a = [a]
f a = c -> a
f2 <*> f1 = \x -> f2 x $ f1 x
-- const id <*> f1 == f1
```

#### 反例

f = Const c -- 常函子



## 半群与应用函子

### Const 应用函子

Const a 在半群意义下是应用函子.

```haskell
-- Const a
instance Monoid a => Applicative (Const a) where
    pure _ = Const mempty :: Const a
    (Const x) <*> (Const y) = (Const x <> y)
```

$x \circledast y =xy$.



### 选择应用函子

类似于加法半群

```haskell
-- Monoid a => [a]
mempty = []
mappend = ++

-- Monoid a => Maybe a
empty = Nothing
Nothing <> x = x <> Nothing = x
Just x <> Just y = Just (x <> y)

-- Alternative
class Applicative f => Alternative f where
    empty :: f a
    <|> :: f a -> f a -> f a
    
instance Alternative [] where
 ...
 
instance Alternative Maybe where
 ...
 
asum :: (Foldable t, Alternative f) => t (f a) -> f a  
asum = foldr (<|>) empty 

class (Alternative m, Monad m) => MonadPlus m where
   mzero :: m a
   mzero = empty
   mplus :: m a -> m a -> m a
   mplus = (<|>)
```

### 拉链应用函子

```haskell
newtype Ziplist = ZipList (getZipList :: [a])
instance Applicative ZipList where
    pure x = ZipList (repeat x)
    ZipList fs <*> ZipList xs = ZipList (zipWith ($) fs xs)
```

$\eta(x)=\{x_i=x\}; Z(\{f_i\})\circledast Z(\{x_i\})= Z(\{f_i(x_i)\})$.



## 单子

单子是Haskell最有特点的一部分

```haskell
class Monad m where  
    return :: a -> m a  -- pure
    join :: m (m a) -> m a -- bind function

    (>>=) :: m a -> (a -> m b) -> m b
    x >>= f = join $ fmap f x

    (>>) :: m a -> m b -> m b  
    x >> y = x >>= \_ -> y  -- mempty >>= f

    fail :: String -> m a  
    fail msg = error msg  
    
(<=<) :: (Monad m) => (b -> m c) -> (a -> m b) -> (a -> m c)  
f <=< g = (\x -> g x >>= f)  -- g >=> f

f <*> x = join $ fmap (\f -> fmap f x) f -- f>>= (\f -> f <$> x)

-- example
-- guard : true|->[()], false|->[]
Prelude Control.Monad> [1..50] >>= (\x -> guard ('7' `elem` show x) >> return x)   
[7,17,27,37,47]  
Prelude Control.Monad> do { x <- [1..50]; guard ('7' `elem` show x); return x }  
[7,17,27,37,47]  
Prelude Control.Monad> [ x | x <- [1..50], '7' `elem` show x ]  
[7,17,27,37,47] 

-- Control.Monad
when/until/void
```



$\triangleright: Ma\to(a\to Mb)\to Mb=(a\to Mb)\to (Ma\to Mb)$

$x\triangleright f=\mu(Mf(x))$.  *Kleisli 求值*

左右单位性 $\eta(x)\triangleright f=f(x), m\triangleright \eta=m$.

结合性 $m\triangleright (\lambda x. f(x)\triangleright g)=m\triangleright f\triangleright g=m\triangleright (f \ast_K g)$ *Kleisli 复合*

$m \gg y=m\triangleright x\mapsto y$ *Kleisli 结合运算*

$c (x\gg y) = c x *_K c y$ *Kleisli 同态*

```haskell
List, join = concat
Maybe, join: Maybe Maybe a -> Maybe a

r ->, join: (r -> r -> a) -> (r -> a)
join f = \x -> f x x

x >>= f = \t -> f(x(t), t)
```



| 函子           | 应用函子            | 单子      |
| -------------- | ------------------- | --------- |
| 解包-计算-打包 | 解包-解包-计算-打包 | 解包-计算 |



### do 语法糖

```haskell
count:: Int
count = sum $ [1..10] >>= (\x -> [x..10] >>= \_ -> return 1)
      = sum $ [1..10] >>= (\x -> [x..10]) >> [1]
count = sum $ do
    x <- [1..10]
    _ <- [x..10]
    return 1

getLine :: IO String
putStrLn :: String -> IO ()
main :: IO ()
main = getLin >>= putStrLn
main = do
    line <- getLine
    putStrLn line

do {x<-s; y<-t; g} == s >>= \x -> (t >>= \y->g) == s >>= ((\x -> t) >=> g)
do {s; t; g} == s >> t >> g
```

 

### 列表单子/控制结构

```haskell
sequence :: t (m a) -> m (t a)  -- 类型转换
-- t = []
sequence [] = return []
sequence a:as = a >>= \x -> ((\l -> x:l) <$> sequence as) -- 类似于 Descartes 积

sequence_ [] = return ()
sequence_ a:as = a >> sequence_ as

mapM = sequence . map :: (a -> m b) -> [a] -> m [b]
mapM_ = sequence_ . map :: (a -> m b) -> [a] -> m ()

forM = flip mapM :: [a] -> (a -> m b) -> m [b] -- 模拟 for 循环
forM_ = flip mapM_

main = forM_ [5..10] $ \n -> do
    putStrLn $ "Solutions for queen" ++ (show n) ++ " problem:"
    forM_ (queensN n) $ print

replicateM :: Int -> m a -> m [a]
replicateM_ :: Int -> m a -> m ()

-- m: s -> a , s
sequence_ [m1, m2, ....] ~ m1'm2'.... :: s -> (), s
replicateM_ n m = m ^ n :: s -> (), s
sequence [m1, m2, ....] ~ m1'm2'.... :: s -> [....], s


forever :: m a -> m b -- sequence . repeat

main = forever $ do
    input <- getLine
    putStrLn $ answer input

filterM :: (a -> m Bool) -> [a] -> m [a]

foldM :: (b -> a -> m b) -> b -> t a -> m b
-- do
-- b1 <- f(b, a0)
-- b2 <- f(b1, a1)
-- ...
-- f(bn, an)
-- == (f(a0)*f(a1)*f(a2)...f(an))(b)

-- example

\m -> mapM . (const m) :: m b -> [a] -> m [b]

```



### State 单子

模拟命令式语言

```haskell
newtype State a s = State {runState: s -> (a, s)}
instance Functor (State s) where
    fmap f fs = State $ \s -> let (a, s') = runState fs s in (f a, s')
    -- (f(a), s'), (a, s') = fs(s)
    
instance Monad (State s) where
    return x = State $ \s -> (x, s)
    fa >>= f = State $ \s -> let (a, s') = runState fa s in (runState (f a) s')
    -- f(a)(s'), (a, s') = fa(s)
    -- do {b <- fa; f b}

    fa >> f == State $ \s -> let (a, s') = runState fa s in (runState f s')
    -- f(s'), (a, s') = fa(s) -- 不利用fa的输出
    -- f . fa_2

get = State $ \s -> (s, s)  -- 获取当前状态，不做改变
put :: s -> State s ()
put s = State $ \_ -> ((), s)
modify f = State $ \s -> ((), f s) -- 无返回值的状态迁移

fa >> put s = State $ \s -> ((), fa_1 s) = modify fa_1
```



### IO 单子

定义 IO 单子的动机

```haskell
-- 第一种定义，foo, bar编译器会认为是一样的
putStrLn' :: String -> ()
getLine' :: String

foo = putStrLn' getLine'
bar = putStrLn' getLine'

-- 第二种定义，执行部分顺序
putStrLn' :: String -> Unique
getLine':: Unique -> String

-- 第三种定义, 变量上的依赖，有求值顺序
foo :: Unique
foo = putStrLn' $ getLine' ???
bar = putStrLn' $ getLine' foo

-- 标准定义
newtype IO a = IO (RealWorld -> (RealWorld, a))

-- 常用函数
getChar:: IO Char
getLine :: IO String
getContents :: IO String
interact :: (String -> String) -> IO ()

readIO :: Read a => String -> IO a
readLn :: Read a => IO a
readMaybe :: Read a => String -> Maybe a

-- 文件操作
readFile :: FilePath -> IO ()
writeFile :: FilePath -> String -> IO()
appendFile :: ...

-- 变量
data IORef a = ...
newIORef :: a -> IORef a  -- a <- newIORef 0 :: IORef Int
readIORef :: IORef a -> IO a
writeIORef :: IORef a -> a -> IO ()
modifyIORef :: IORef a -> (a->a) -> IO ()  -- modifyIORef‘ 立刻执行
```

IO 操作自动不是阻塞的，所有读写操作都在运行时交给IO管理器处理。



### ST 单子— 状态线程单子，比IO弱的单子

```haskell
newtype ST s a = ST (STRef s a)
type STRef s a = State# s -> (# State# s, a #)
-- newtype ST s a = ST (s -> (s, a))

newSTRef :: a -> ST s (STRef s a)
readSTRef :: STRef s a -> ST s a
writeSTRef :: STRef s a -> a -> ST s ()
modifySTRef

-- example
fib :: Int -> ST s Integral
fib n = do {a <- newSTRef 0;
            b <- newSTRef 1;
            repeatFor n
            (do {x <- readSTRef a;
                y <- readSTRef b;
                writeSTRef a y;
                writeSTRef b $! (x+y)});
            readSTRef a}
```



#### 可变数组

```haskell
newListArray :: Ix i => (i, i) -> [e] -> ST s (STArray s i e)
getElems :: STArray s i e -> ST s [e]
```



### Reader 单子

```haskell
-- 模板渲染

headT, bodyT :: String -> String  -- Template

-- greatingMike = headT "Mike" ++ bodyT "Mike"
renderGreeting = gather <$> headT <*> bodyT <*> where
    gather x y z = x ++ y
    
data Greet = Greet {
    headT :: String
    , bodyT :: String
} deriving Show

renderGreeting :: String -> Greet
renderGreeting = Greet <$> headT <*> bodyT  -- Greet (headT x)  (bodyT x)

renderGreeting = do
    h <- headT
    b <- bodyT
    return $ Great h b

--
data Greet = Greet {
    name :: String
    , headT :: String
    , bodyT :: String
    , footT :: String
} deriving Show

renderGreeting = do
    n <- ask
    h <- headT
    (b, f) <- local ("Mr. and Mrs." ++) $ do
        b' <- bodyT
        f' <- footT
        return (b', f')  -- x -> (bodyT(x), footT(x)) : String -> (String, String)
    return $ Greet n h b f

-- renderGreeting(x) = Greet(n(x), h(x), b(l(x)), f(l(x)))

-- Reader

newtype Reader r a = Reader {runReader :: r -> a}
instance Functor (Reader r) where
    fmap f m = Reader $ \r -> f (runReader m r) = Reader (f . (runReader m))

m >> f = Reader $ \r -> runReader (f (runReader m r)) r
```

$Rr:$ Reader 函子, $Hom(r,\cdot)$

$Rr(f)x= R(\lambda y:r. f(\phi(x)(y)))=R(f\circ\phi(x))$

$Rr(f)R(g)=R(fg)$.

$\mu(x)=R(\lambda t:r.\phi(\phi(x)(t))(t)), \mu(Rg)=R(\lambda t:r.\phi(g(t))(t)).$



### 半群作用单子

```haskell
f a = (m, a)
(x, a) >>= f = (xf(a)1, f(a)2)
```

$\phi((s,a), f)=(sf(a)_1, f(a)_2)$.

$\mu(s_1,s_2,a)=(s_1s_2,a),\eta(a)=(1,a)$.



### 半群State单子

```haskell
f a = a->(m, a)
```

$f\star g = f \triangleright \lambda m.F(m) g, f\star g (a)=\lambda m. (m g(s)_1, g(s)_2), (m,s)=f(s)$.



## Traversable

### Foldable

```haskell
class Foldable t where
    fold :: Monoid m => t m -> m
    foldMap :: Monoid m => (a -> m) -> t a -> m
    foldl/foldr
    foldl'/foldr' -- 严格求值
    -- foldMap f = fold . (<$> f)
    
    
data BinaryTree a = Nil | Node a (BinaryTree a) (BinaryTree a)
     deriving (Show)
     
instance Foldable BinaryTree where
      foldr f acc Nil = acc
      foldr f acc (x:xs)
      = (foldr f (f x (foldr f acc right)) left)
      
      foldMap f Nil = mempty
      foldMap f (Node x left right)
          = foldMap f left `mappend` f x `mappend` foldMap f right
 
 foldr f z t = appEndo (foldMap (Endo #. f) t) z
 foldl f z t = appEndo (getDual (foldMap (Dual . Endo . filp f) t)) z
```

$F(f,\{a_\lambda\})=\prod_\lambda f(a_\lambda), f:a\to S,\{a_\lambda\}:ta$，半群折叠形式

$d(f, z, t)=F(f,\{a_\lambda\})(z),f:a\to b\to b=a\to Endo(b)$, 折叠reduce形式



### Traversable

```haskell

 instance Traversable BinaryTree where
     traverse f Nil = pure Nil
     traverse f (Node x left right) =
         Node <$> f x <*> traverse f left <*> traverse f right
     -- defined by fmap
     traverse f t = mfo (fmap f t) where
     mfo :: BinaryTree (f b) -> f (BinaryTree b)  -- sequenceA
     mfo Nil = Nil
     mfo (Node x left right) =
         Node <$> x <*> mfo left <*> mfo right
         
mapM_ f = foldr ((>>).f) (return ())
```

$T(f, N(x, L, R))=F(N)(f(x), T(f,L), T(f, R)):(a\to Fb) \to (Ta \to FTb)$.

$S(N)=T(i, N): TFa\to FTa, T(f,N)=S(\tilde{f}(N))$.



## 单子变换: 构造函数 m -> tm

$t:m\mapsto tm, *\to*\to*\to*$

$t$ 在$m$上添加新的功能

### ReaderT

```haskell
newtype ReaderT r m a = ReaderT {runReaderT :: r-> m a}
-- ReaderT r m a = r -> m a

instance (Monad m) => Monad (ReaderT r m) where
m >>= k = ReaderT $ \r -> do
    a <- runReaderT m r
    runReaderT (k a) r
-- ReaderT $ \r -> runReaderT m r >>= \a -> runReaderT (k a) r
-- \r -> m(r) >>= \a -> k(a)(r), r represents the local information

liftReaderT :: m a -> ReaderT r m a
liftReaderT m = ReaderT (const m)
-- ReaderT \_ -> m

-- example
printEnv :: ReaderT String IO ()
printEnv = do
    ReaderT $ \env -> putStrLn ("Here's " ++ env)
    
ask :: Monad m => ReaderT r m r
ask = ReaderT return

local :: (r -> r) -> ReaderT r m a -> ReaderT r m a
local f m = ReaderT $ \r -> runReaderT m (f r)
```



#### StateT

```haskell
newtype StateT s m a = State {runStateT :: s -> m (a, s)}

instance (Monad m) => Monad (StateT s m) where
    m >>= k = StateT $ \s -> do
    (a, s') <- runState m s
    runStateT (k a) s'
    
    liftState :: (Monad m) => m a -> StatT s m a
    liftState m = StateT $ \s -> do
        a <- m
        return (a, s)
        
    state f = StateT return . f  -- eta.f
```

$ Ta = s \to M (a\times s), \eta(a) = \lambda s. \eta(a,s)$

$m\triangleright k = \lambda s. m(s) \triangleright k, l(m)=\lambda s. m\triangleright \lambda a.\eta(a,s)$

```haskell
lift p :: StateT s IO a -- s -> w -> ((a,s), w'), p: w -> (a, w') ; s-fixed
st f = return . f -- s -> w -> (f(s), w)   ; w-fixed

lift p -- p:w->(a,w)=Sa  s->w->((a, s), w'); s,w-fixed p is a reading method
```



### MonadTrans

```haskell
class MonadTrans t where
    lift :: Monad m => m a -> t m a
    
instance MonadTrans (ReaderT r) where
    lift = liftReaderT
    
instance MonadTrans (StateT r) where
    lift = liftStateT
   
newtype MaybeT m a = MaybeT {runMaybeT :: m (Maybe a)}
instance MonadTrans MaybeT where
    lift m = MaybeT m :: m a -> MaybeT m a
    
-- lift . return =return
-- lift (m >>= k) = (lift m) >>= (lift . k)

newtype RandT g m a = RandT (StatT g m a)
(Monad m, RandomGen g) => MonadRandom (RandT g m)

-- R g m a == g -> m (a, g)

```



### WriterT

```haskell
newtype WriterT w m a = WriterT {runWriterT :: m (a, w)}
instance (Monoid w, Monad m) => Monad (WriterT w m) where
    return a = writer (a, empty)
    m >>= k = WriterT $ do
    (a, w) <- runWriterT m
    (y, w') <- runWriterT (k, a)
    return (y, w <> w')
```


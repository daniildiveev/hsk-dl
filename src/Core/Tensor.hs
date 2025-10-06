module Core.Tensor
  ( Device(..)
  , Layout(..)
  , Tensor(..)
  , shape, stride, offset
  , numel
  , empty, zeros, ones, full
  , fromList, toList
  , clone, contiguous, isContiguous
  ) where

import Foreign.ForeignPtr (ForeignPtr)

data Device = CPU
  deriving (Eq, Show)

data Layout = Strided
  { _shape  :: ![Int]
  , _stride :: ![Int]
  , _offset :: !Int
  } deriving (Eq, Show)

data Tensor a = Tensor
  { layout  :: !Layout
  , storage :: !(ForeignPtr a)
  , len     :: !Int
  , device  :: !Device
  }

shape :: Tensor a -> [Int]
shape = _shape . layout

stride :: Tensor a -> [Int]
stride = _stride . layout

offset :: Tensor a -> Int
offset = _offset . layout

numel :: [Int] -> Int
numel [] = 1
numel ds = product ds

empty :: [Int] -> IO (Tensor a)
empty _ = error "NYI: empty"

zeros :: [Int] -> IO (Tensor a)
zeros _ = error "NYI: zeros"

ones :: [Int] -> IO (Tensor a)
ones _ = error "NYI: ones"

full :: [Int] -> a -> IO (Tensor a)
full _ _ = error "NYI: full"

fromList :: [Int] -> [a] -> IO (Tensor a)
fromList _ _ = error "NYI: fromList"

toList :: Tensor a -> IO [a]
toList _ = error "NYI: toList"

clone :: Tensor a -> IO (Tensor a)
clone _ = error "NYI: clone"

contiguous :: Tensor a -> IO (Tensor a)
contiguous _ = error "NYI: contiguous"

isContiguous :: Tensor a -> Bool
isContiguous _ = False

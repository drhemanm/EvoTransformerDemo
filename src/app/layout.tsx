import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EvoCompliance",
  description: "Compliance intelligence powered by EvoTransformer",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
